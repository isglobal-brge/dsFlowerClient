# Module: Unified submission (researcher side)
#
# The single researcher-facing path that turns a model spec into a running
# federated DP job. There is NO server-side catalog: the client codegens the
# submission (an nn.Module package, or an XGBoost spec), ships it in ONE FAB
# (ServerApp + ClientApp + the model), the nodes spool it for the run, enforce DP
# node-side (DP-SGD / DP-GBDT / egress gate per the manifest dp-track), and delete
# it afterward.

#' @keywords internal
.runner_skeleton_dir <- function() {
  p <- system.file("flower_app", "dsflower_runner", package = "dsFlowerClient")
  if (nzchar(p) && dir.exists(p)) return(p)
  alt <- file.path("inst", "flower_app", "dsflower_runner")
  if (dir.exists(alt)) return(normalizePath(alt))
  stop("Bundled runner (dsflower_runner) not found.", call. = FALSE)
}

#' Build the submission FAB: the trusted runner skeleton + (neural/egress) the
#' user package + a pyproject carrying the dp-track and run config.
#' @keywords internal
.build_submission_app <- function(sub, config_lines, results_dir) {
  app_dir <- file.path(tempdir(), "dsflower_submission", "dsflower-app")
  if (dir.exists(app_dir)) unlink(app_dir, recursive = TRUE)
  dir.create(app_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(.runner_skeleton_dir(), app_dir, recursive = TRUE)

  packages <- "dsflower_runner"
  if (!is.null(sub$pkg_dir)) {
    file.copy(normalizePath(sub$pkg_dir), app_dir, recursive = TRUE)
    unlink(file.path(app_dir, sub$package, "__pycache__"), recursive = TRUE)
    packages <- c(packages, sub$package)
  }
  pkg_toml <- paste0('["', paste(packages, collapse = '", "'), '"]')

  toml <- paste0(
    '[build-system]\nrequires = ["hatchling"]\n',
    'build-backend = "hatchling.build"\n\n',
    '[project]\nname = "dsflower-app"\nversion = "1.0.0"\n',
    'description = "dsFlower submission"\nlicense = "MIT"\n',
    'dependencies = [', paste0('"', .harness_dependencies(), '"', collapse = ", "), ']\n\n',
    '[tool.hatch.build.targets.wheel]\npackages = ', pkg_toml, '\n\n',
    '[tool.flwr.app]\npublisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "dsflower_runner.server_app:app"\n',
    'clientapp = "dsflower_runner.client_app:app"\n\n',
    '[tool.flwr.app.config]\n', paste(config_lines, collapse = "\n"), '\n'
  )
  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  app_dir
}

#' Submit + run a federated DP job from a model spec (the "pack" API)
#'
#' Codegens the submission, ships it in one FAB to the DataSHIELD SuperNodes,
#' runs it under the manifest-pinned enforced-DP track, and tears it down.
#'
#' @param conns DSI connections.
#' @param model A model name or \code{dsflower_model} (registry-resolved).
#' @param target Character; target column.
#' @param features Character vector; feature columns.
#' @param symbol Character; server-side data handle symbol.
#' @param privacy A \code{dsflower_privacy} (default \code{ds.flower.privacy()}).
#' @param num_rounds Integer; federated rounds.
#' @param model_params Named list; params merged over the model's defaults.
#' @param data_kind "tabular" or "image".
#' @param verbose Logical.
#' @return A \code{dsflower_run}.
#' @export
ds.flower.submit <- function(conns, model, target, features = NULL,
                             data = NULL, resource = NULL, symbol = NULL,
                             privacy = NULL, num_rounds = 1L,
                             model_params = list(), data_kind = "tabular",
                             torch_backend = "auto", verbose = TRUE) {
  .require_flwr_cli()
  privacy <- .resolve_privacy(privacy)
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  if (length(model_params)) {
    model$params <- utils::modifyList(model$params %||% list(), model_params)
  }
  sub <- .emit_submission(model)

  if (is.null(data) && is.null(resource) && is.null(symbol)) symbol <- "D"
  flower <- ds.flower.connect(conns, data = data, resource = resource, symbol = symbol)
  conns <- flower$conns; hsym <- flower$symbol
  n_clients <- length(conns)
  n_features <- if (!is.null(features)) length(features) else 0L

  up <- NULL
  if (!is.null(sub$pkg_dir)) {
    up <- .upload_user_module(conns, sub$pkg_dir)
    if (verbose) message("  Model package '", up$package, "' uploaded + scanned.")
  }

  # The server-side disclosure class-distribution check applies only to
  # classification; a continuous regression / count target legitimately has ~1 row
  # per value. Derive the task type from the pinned loss so the check is skipped
  # correctly (the node also skips its label-range assertion for these losses).
  task_type <- if (identical(sub$track, "trees")) "classification" else
    switch(sub$loss %||% "bce_logits",
           mse = "regression", poisson_nll = "count", negbin_nll = "count",
           gamma_nll = "regression", "classification")
  # Propagate data_type so the node stages a table-backed IMAGE collection (samples
  # table + dsflower.image_data_root) as image, not tabular. dsImaging resources
  # self-declare image via their descriptor; a plain samples table needs this signal.
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = list("task-type" = task_type, "dp-track" = sub$track,
                      "data_type" = data_kind),
    privacy = privacy)

  if (!is.null(up)) {
    pin <- DSI::datashield.aggregate(conns, call("flowerTier2PinDS", hsym, up$token))
    if (verbose) message("  Pinned: ",
                         paste(unique(unlist(lapply(pin, `[[`, "pinned"))), collapse = ", "), ".")
  }

  results_dir <- file.path(tempdir(), "dsflower_results",
                           format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cfg <- c(
    .toml_kv("dp-track", sub$track),
    .toml_kv("data-kind", data_kind),
    paste0("num-features = ", as.integer(n_features)),
    paste0("num-server-rounds = ", as.integer(num_rounds)),
    paste0("min-train-nodes = ", as.integer(n_clients)),
    .toml_kv("results-dir", results_dir)
  )
  if (identical(sub$track, "neural")) {
    p <- sub$params
    cfg <- c(cfg,
      .toml_kv("model-spec-b64", .spec_to_b64(sub$spec)),
      .toml_kv("loss-name", sub$loss %||% "bce_logits"),
      paste0("num-classes = ", as.integer(p$n_classes %||% p$num_classes %||% 2L)),
      paste0("num-labels = ", as.integer(p$num_labels %||% 2L)),
      paste0("local-epochs = ", as.integer(p$local_epochs %||% 1L)),
      paste0("batch-size = ", as.integer(p$batch_size %||% 32L)),
      paste0("learning-rate = ", as.numeric(p$learning_rate %||% 0.01)),
      paste0("nb-dispersion = ", as.numeric(p$nb_dispersion %||% 1.0)),
      paste0("gamma-shape = ", as.numeric(p$gamma_shape %||% 1.0)))
    if (identical(data_kind, "image")) {
      cfg <- c(cfg, .toml_kv("backbone", as.character(p$backbone %||% "resnet18")),
               paste0("image-size = ", as.integer(p$image_size %||% 224L)))
    }
  } else if (identical(sub$track, "trees")) {
    s <- sub$spec
    cfg <- c(cfg,
      .toml_kv("objective", as.character(s$objective)),
      paste0("max-depth = ", as.integer(s$max_depth)),
      paste0("n-trees = ", as.integer(s$n_trees)),
      paste0("learning-rate = ", as.numeric(s$learning_rate)),
      paste0("reg-lambda = ", as.numeric(s$reg_lambda)),
      paste0("n-bins = ", as.integer(s$n_bins)))
  }

  app_dir <- .build_submission_app(sub, cfg, results_dir)
  .ensure_client_framework("pytorch")

  # link.up starts the INSECURE local SuperLink (the bytes already travel inside
  # the TLS DataSHIELD channel) AND the per-node DSI tunnel forwarders. Do NOT
  # start the SuperLink separately: a default SSL SuperLink mismatches the
  # insecure inner gRPC the SuperNodes speak and the run hangs.
  ds.flower.link.up(conns)
  recipe <- structure(list(
    model = list(framework = "pytorch", track = sub$track),
    strategy = list(name = "FedAvg", params = list()),
    privacy = privacy, num_rounds = as.integer(num_rounds),
    features = features, evaluation_only = FALSE), class = "dsflower_recipe")

  run <- tryCatch({
    ds.flower.nodes.ensure(conns, hsym, torch_backend = torch_backend)
    ds.flower.run.start(recipe, conns, app_dir = app_dir,
                        results_dir = results_dir, symbol = hsym, verbose = verbose)
  }, error = function(e) {
    tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
    tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
    if (!is.null(up)) tryCatch(DSI::datashield.aggregate(conns, call("flowerAppDeleteDS", up$token)), error = function(e) NULL)
    stop(e)
  })

  tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
  tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
  if (!is.null(up)) tryCatch(DSI::datashield.aggregate(conns, call("flowerAppDeleteDS", up$token)), error = function(e) NULL)
  tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  run
}


# --------------------------------------------------------------------------- #
# Shared FAB helpers (TOML formatting + the trusted app's pip dependencies),
# used by .build_submission_app here and .build_tier2_app (egress track).
# --------------------------------------------------------------------------- #

#' Format a key-value pair for TOML
#' @keywords internal
.toml_kv <- function(key, val) {
  if (is.character(val)) {
    paste0(key, ' = "', val, '"')
  } else if (is.logical(val)) {
    paste0(key, " = ", tolower(as.character(val)))
  } else if (is.integer(val) && length(val) == 1) {
    paste0(key, " = ", val)
  } else if (is.numeric(val) && length(val) == 1) {
    paste0(key, " = ", val)
  } else if (is.numeric(val) || is.integer(val)) {
    paste0(key, " = [", paste(val, collapse = ", "), "]")
  } else {
    paste0(key, ' = "', as.character(val), '"')
  }
}

#' Pip dependencies for the trusted app (one app, not per-model). Vision adds
#' torchvision + the medical-image readers (2D, plus the MONAI 3D path).
#' @keywords internal
.harness_dependencies <- function(vision = FALSE) {
  base <- c("flwr[app]>=1.31.0", "numpy>=1.21.0", "pandas>=1.3.0",
            "torch>=2.0.0", "opacus>=1.4.0")
  if (!isTRUE(vision)) return(base)
  c(base, "torchvision>=0.15.0", "pillow>=9.0.0",
    "nibabel>=5.0.0", "pynrrd>=1.0.0", "SimpleITK>=2.2.0", "monai>=1.3.0")
}
