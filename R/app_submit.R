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

#' Compute the canonical hash of the bundled runner
#'
#' This must remain byte-identical to dsFlower's .compute_harness_hash():
#' radix-sorted relative path, newline, content, NUL; compiled files excluded.
#' @keywords internal
.compute_local_runner_hash <- function(pkg_dir = .runner_skeleton_dir()) {
  rel_files <- list.files(pkg_dir, recursive = TRUE, full.names = FALSE,
                          all.files = TRUE, no.. = TRUE)
  rel_files <- rel_files[!grepl("(^|/)__pycache__(/|$)", rel_files)]
  rel_files <- rel_files[!grepl("\\.(pyc|pyo)$", rel_files)]
  rel_files <- sort(rel_files, method = "radix")
  blob <- raw(0)
  for (rel in rel_files) {
    full <- file.path(pkg_dir, rel)
    content <- readBin(full, "raw", file.info(full)$size)
    blob <- c(blob, charToRaw(rel), charToRaw("\n"), content, as.raw(0x00))
  }
  digest::digest(blob, algo = "sha256", serialize = FALSE)
}

#' Fail early when a node cannot execute this client's trusted runner
#' @keywords internal
.assert_runner_compatibility <- function(conns, symbol) {
  expected_hash <- .compute_local_runner_hash()
  caps <- tryCatch(
    DSI::datashield.aggregate(
      conns, expr = call("flowerGetCapabilitiesDS", symbol)),
    error = function(e) {
      stop("Could not verify the dsFlower runner compatibility: ",
           conditionMessage(e), call. = FALSE)
    }
  )
  if (!is.list(caps) || !length(caps)) {
    stop("Could not verify the dsFlower runner compatibility: no node capabilities returned.",
         call. = FALSE)
  }

  node_names <- names(caps)
  if (is.null(node_names)) node_names <- rep("", length(caps))
  missing_names <- !nzchar(node_names)
  node_names[missing_names] <- paste0("node", which(missing_names))
  failures <- character()
  for (i in seq_along(caps)) {
    cap <- caps[[i]]
    abi <- if (is.list(cap)) {
      tryCatch(suppressWarnings(as.integer(cap[["runner_abi"]])),
               error = function(e) integer())
    } else integer()
    remote_hash <- if (is.list(cap)) {
      tryCatch(tolower(as.character(cap[["runner_sha256"]])),
               error = function(e) character())
    } else character()
    abi_ok <- length(abi) == 1L && !is.na(abi) && identical(abi, 2L)
    hash_ok <- length(remote_hash) == 1L && !is.na(remote_hash) &&
      identical(remote_hash, expected_hash)
    if (!abi_ok || !hash_ok) {
      abi_label <- if (length(abi) == 1L && !is.na(abi)) as.character(abi) else "missing"
      hash_label <- if (length(remote_hash) == 1L && !is.na(remote_hash) &&
                        nzchar(remote_hash)) substr(remote_hash, 1L, 12L) else "missing"
      failures <- c(failures, paste0(
        node_names[[i]], " (runner_abi=", abi_label,
        ", runner_sha256=", hash_label, ")"))
    }
  }
  if (length(failures)) {
    stop("Incompatible dsFlower runner on ", paste(failures, collapse = ", "),
         "; expected runner_abi=2 and runner_sha256=",
         substr(expected_hash, 1L, 12L),
         ". Update dsFlower/dsFlowerClient so their bundled runners match.",
         call. = FALSE)
  }
  invisible(caps)
}

#' Build the submission FAB: the trusted runner skeleton + (neural/egress) the
#' user package + a pyproject carrying the dp-track and run config.
#' @keywords internal
.build_submission_app <- function(sub, config_lines, results_dir, vision = FALSE) {
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
    'dependencies = [', paste0('"', .harness_dependencies(vision), '"', collapse = ", "), ']\n\n',
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
#' @param data Optional character data source resolved during connection.
#' @param resource Optional Opal resource name.
#' @param symbol Character; server-side data handle symbol.
#' @param num_rounds Integer; federated rounds.
#' @param feature_bounds Optional public feature bounds as
#'   \code{list(lower=..., upper=...)} in feature order. These constants are
#'   supplied without querying node data and define a clipped affine transform.
#' @param target_levels Optional ordered, exhaustive public label vocabulary for
#'   classification. Non-numeric targets require it; node values are never used
#'   to infer label codes.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count targets. The node clips each target to these constants.
#' @param model_params Named list; params merged over the model's defaults.
#'   Neural and DP-GBDT learning rates must be finite and in \code{(0, 10]}.
#' @param data_kind "tabular" or "image".
#' @param strategy Character; researcher-side aggregation strategy.
#' @param output_dir Optional model output directory.
#' @param output_name Optional model output name.
#' @param torch_backend Character; node torch backend selection.
#' @param verbose Logical.
#' @param silent Logical; suppress progress feedback.
#' @return A \code{dsflower_run}.
#' @export
ds.flower.submit <- function(conns, model, target, features = NULL,
                             data = NULL, resource = NULL, symbol = NULL,
                             num_rounds = 1L,
                             model_params = list(), data_kind = "tabular",
                             strategy = "fedavg",
                             output_dir = NULL, output_name = NULL,
                             torch_backend = "auto", verbose = FALSE,
                             silent = FALSE, feature_bounds = NULL,
                             target_levels = NULL, target_bounds = NULL) {
  rounds_value <- suppressWarnings(as.numeric(num_rounds))
  if (length(rounds_value) != 1L || is.na(rounds_value) || !is.finite(rounds_value) ||
      rounds_value < 1 || rounds_value %% 1 != 0 ||
      rounds_value > .Machine$integer.max) {
    stop("num_rounds must be a single positive integer.", call. = FALSE)
  }
  num_rounds <- as.integer(rounds_value)
  .require_flwr_cli()
  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  if (length(model_params)) {
    model$params <- utils::modifyList(model$params %||% list(), model_params)
  }
  sub <- .emit_submission(model)
  if (identical(sub$track, "neural")) {
    lr <- suppressWarnings(as.numeric((sub$params %||% list())$learning_rate %||% 0.01))
  } else if (identical(sub$track, "trees")) {
    lr <- suppressWarnings(as.numeric((sub$spec %||% list())$learning_rate %||% 0.3))
  } else {
    lr <- 0.01
  }
  if (length(lr) != 1L || is.na(lr) || !is.finite(lr) || lr <= 0 || lr > 10) {
    stop("learning_rate must be one finite value in (0, 10].", call. = FALSE)
  }

  # Aggregation strategy. All of these run ONLY on the researcher-side SuperLink,
  # over the already-DP client updates -> pure post-processing, so the (epsilon,
  # delta) guarantee is unchanged whichever is chosen (it is never a privacy knob).
  strategy <- tolower(as.character(strategy)[1])
  .dsf_strategies <- c("fedavg", "fedadam", "fedadagrad", "fedyogi", "fedavgm")
  if (!strategy %in% .dsf_strategies) {
    stop("Unknown strategy '", strategy, "'. Supported: ",
         paste(.dsf_strategies, collapse = ", "), ".", call. = FALSE)
  }
  if (identical(sub$track, "trees") && !identical(strategy, "fedavg")) {
    warning("strategy = '", strategy, "' is ignored on the trees track; ",
            "boosted trees aggregate boosters, not averaged weights.",
            call. = FALSE)
    strategy <- "fedavg"
  }

  if (is.null(data) && is.null(resource) && is.null(symbol)) symbol <- "D"

  # Auto-detect tabular features = every column except the target(s), from the DATA symbol's
  # schema (the assigned data.frame -- NOT the flower handle). Column NAMES are schema, not
  # values, so the standard DataSHIELD `colnames` aggregate is non-disclosive. Image runs
  # derive their feature dim node-side; symbol-less (data/resource) advanced calls must pass
  # features explicitly.
  if (is.null(features) && !is.null(symbol) && !identical(data_kind, "image")) {
    cols <- tryCatch(DSI::datashield.aggregate(conns, call("colnamesDS", symbol)),
                     error = function(e) NULL)
    if (!is.null(cols) && length(cols) && length(cols[[1L]])) {
      features <- setdiff(as.character(cols[[1L]]), as.character(target))
    }
    if (is.null(features) || length(features) == 0L) {
      stop("Could not auto-detect feature columns for symbol '", symbol,
           "'. Pass `features` explicitly.", call. = FALSE)
    }
  }
  # NEVER let the target leak in as a feature (auto-detect excludes it; an explicit list might
  # include it by mistake -> the model would train on the answer). Drop it defensively.
  if (!is.null(features)) {
    features <- as.character(features)
    dropped <- intersect(features, as.character(target))
    if (length(dropped)) {
      features <- setdiff(features, as.character(target))
      message("  Dropped target column(s) from features: ", paste(dropped, collapse = ", "))
    }
    if (anyDuplicated(features)) {                       # de-dupe: avoid silent duplicate inputs
      features <- unique(features)
    }
  }

  # No exact statistics leave a data node.  Optional scaling/ranges are public
  # constants supplied by the analyst before the run; DP-SGD still clips every
  # per-sample gradient independently of this utility-only transform.
  public_bounds <- .validate_public_feature_bounds(feature_bounds, features)
  feature_norm <- if (is.null(public_bounds)) NULL else list(
    means = (public_bounds$lower + public_bounds$upper) / 2,
    sds = (public_bounds$upper - public_bounds$lower) / 2)
  if (!is.null(public_bounds) && verbose) {
    message("  Preprocessing: using public bounds for ", length(features),
            " feature(s); no node statistics were queried.")
  }

  task_type <- if (identical(sub$track, "trees")) "classification" else
    switch(sub$loss %||% "bce_logits",
           mse = "regression", poisson_nll = "count", negbin_nll = "count",
           gamma_nll = "regression", "classification")
  n_target_classes <- if (identical(sub$track, "trees")) 2L else {
    p <- sub$params %||% list()
    as.integer(p$n_classes %||% p$num_classes %||% 2L)
  }
  public_target <- .validate_public_target_spec(
    target_levels, target_bounds, task_type = task_type,
    loss_name = sub$loss %||% "", n_classes = n_target_classes)

  flower <- ds.flower.connect(conns, data = data, resource = resource, symbol = symbol)
  conns <- flower$conns; hsym <- flower$symbol
  n_clients <- length(conns)
  # GUARANTEED cleanup of server-side state on ANY exit (success OR a pre-run error in
  # upload/prepare/pin): drop the tunnel, the node handle/staging, the uploaded app, and the
  # connection. Prevents leaked handles/tokens when prepare fails. up is set by the upload.
  up <- NULL
  on.exit({
    tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
    tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
    if (!is.null(up)) {
      tryCatch(DSI::datashield.aggregate(conns, call("flowerAppDeleteDS", up$token)),
               error = function(e) NULL)
    }
    tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  }, add = TRUE)
  .assert_runner_compatibility(conns, hsym)
  # Tabular runs need a non-empty feature set. The symbol path auto-detects above; data=/
  # resource= inputs must pass `features` explicitly -- fail with a clear message rather than
  # the downstream "num-features must be set" from the node.
  if ((is.null(features) || length(features) == 0L) && !identical(data_kind, "image")) {
    stop("No feature columns: pass `features` explicitly (auto-detect is only available for ",
         "the `symbol=` data path).", call. = FALSE)
  }
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
  # Propagate data_type so the node stages a table-backed IMAGE collection (samples
  # table + dsflower.image_data_root) as image, not tabular. dsImaging resources
  # self-declare image via their descriptor; a plain samples table needs this signal.
  prepare_config <- list(
    "task-type" = task_type,
    "dp-track" = sub$track,
    "data_type" = data_kind,
    "num-server-rounds" = as.integer(num_rounds),
    "num-features" = as.integer(n_features))
  if (identical(sub$track, "neural")) {
    p <- sub$params
    prepare_config <- c(prepare_config, list(
      "model-spec-b64" = .spec_to_b64(sub$spec),
      "loss-name" = sub$loss %||% "bce_logits",
      "num-classes" = as.integer(p$n_classes %||% p$num_classes %||% 2L),
      "num-labels" = as.integer(p$num_labels %||% 2L),
      "local-epochs" = as.integer(p$local_epochs %||% 1L),
      "batch-size" = as.integer(p$batch_size %||% 32L)))
  } else if (identical(sub$track, "trees")) {
    prepare_config[["gbdt-spec"]] <- sub$spec
  }
  if (!is.null(public_bounds)) {
    prepare_config[["feature-bounds"]] <- public_bounds
  }
  if (!is.null(public_target$levels)) {
    prepare_config[["target-levels"]] <- public_target$levels
  }
  if (!is.null(public_target$bounds)) {
    prepare_config[["target-bounds"]] <- public_target$bounds
  }
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = prepare_config)

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
    .toml_kv("strategy", strategy),
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
      paste0("gamma-shape = ", as.numeric(p$gamma_shape %||% 1.0)),
      paste0("weight-decay = ", as.numeric(p$weight_decay %||% 0.0)),
      paste0("l1-penalty = ", as.numeric(p$l1_penalty %||% 0.0)))
    if (!is.null(public_bounds))
      cfg <- c(cfg, .toml_kv("feature-bounds-b64", .spec_to_b64(public_bounds)))
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
    if (!is.null(public_bounds))
      cfg <- c(cfg, .toml_kv("feature-bounds-b64", .spec_to_b64(public_bounds)))
  }

  app_dir <- .build_submission_app(sub, cfg, results_dir,
                                   vision = identical(data_kind, "image"))
  .ensure_client_framework("pytorch")

  # link.up starts the INSECURE local SuperLink (the bytes already travel inside
  # the TLS DataSHIELD channel) AND the per-node DSI tunnel forwarders. Do NOT
  # start the SuperLink separately: a default SSL SuperLink mismatches the
  # insecure inner gRPC the SuperNodes speak and the run hangs.
  ds.flower.link.up(conns)
  recipe <- structure(list(
    model = list(name = model$name, template = model$template %||% model$name,
                 framework = model$framework %||% "pytorch", track = sub$track),
    strategy = list(name = strategy, params = list()),
    num_rounds = as.integer(num_rounds),
    features = features,
    feature_means = if (!is.null(feature_norm)) feature_norm$means else NULL,
    feature_sds   = if (!is.null(feature_norm)) feature_norm$sds   else NULL,
    feature_lower = if (!is.null(public_bounds)) public_bounds$lower else NULL,
    feature_upper = if (!is.null(public_bounds)) public_bounds$upper else NULL,
    target_levels = public_target$levels,
    target_bounds = public_target$bounds,
    evaluation_only = FALSE), class = "dsflower_recipe")

  # Cleanup (tunnel/handle/app/connection) is guaranteed by the on.exit above, on both a
  # run error and normal completion.
  ds.flower.nodes.ensure(conns, hsym, torch_backend = torch_backend)
  ds.flower.run.start(recipe, conns, app_dir = app_dir,
                      results_dir = results_dir, symbol = hsym,
                      output_dir = output_dir, output_name = output_name,
                      verbose = verbose, silent = silent)
}

#' Validate data-independent preprocessing bounds
#' @keywords internal
.validate_public_feature_bounds <- function(bounds, features) {
  if (is.null(bounds)) return(NULL)
  if (is.null(features) || !length(features)) {
    stop("feature_bounds requires an explicit, non-empty feature list.",
         call. = FALSE)
  }
  if (!is.list(bounds) || is.null(bounds$lower) || is.null(bounds$upper)) {
    stop("feature_bounds must be list(lower=..., upper=...).", call. = FALSE)
  }
  lower <- as.numeric(bounds$lower)
  upper <- as.numeric(bounds$upper)
  if (length(lower) != length(features) || length(upper) != length(features)) {
    stop("feature_bounds lower/upper must match the feature count and order.",
         call. = FALSE)
  }
  if (any(!is.finite(lower)) || any(!is.finite(upper)) || any(lower >= upper) ||
      any(abs(lower) > 1e6) || any(abs(upper) > 1e6)) {
    stop("feature_bounds must be finite, inside [-1e6, 1e6], with lower < upper for every feature.",
         call. = FALSE)
  }
  list(lower = lower, upper = upper, features = as.character(features))
}

#' Validate data-independent target semantics
#' @keywords internal
.validate_public_target_spec <- function(levels, bounds, task_type,
                                         loss_name = "", n_classes = 2L) {
  numeric_task <- task_type %in% c("regression", "count", "continuous")
  if (isTRUE(numeric_task)) {
    if (!is.null(levels)) {
      stop("target_levels is only valid for classification targets.",
           call. = FALSE)
    }
    if (!is.list(bounds) || is.null(bounds$lower) || is.null(bounds$upper)) {
      stop("target_bounds=list(lower=..., upper=...) is required for ",
           "regression/count targets.", call. = FALSE)
    }
    lower <- suppressWarnings(as.numeric(bounds$lower))
    upper <- suppressWarnings(as.numeric(bounds$upper))
    if (length(lower) != 1L || length(upper) != 1L ||
        !is.finite(lower) || !is.finite(upper) || lower >= upper ||
        abs(lower) > 1e6 || abs(upper) > 1e6) {
      stop("target_bounds must contain finite scalar lower < upper inside [-1e6, 1e6].",
           call. = FALSE)
    }
    if (identical(task_type, "count") && lower < 0) {
      stop("Count target_bounds require lower >= 0.", call. = FALSE)
    }
    if (identical(loss_name, "gamma_nll") && lower <= 0) {
      stop("gamma_nll target_bounds require lower > 0.", call. = FALSE)
    }
    return(list(levels = NULL, bounds = list(lower = lower, upper = upper)))
  }

  if (!is.null(bounds)) {
    stop("target_bounds is only valid for regression/count targets.",
         call. = FALSE)
  }
  if (is.null(levels)) return(list(levels = NULL, bounds = NULL))
  if (is.factor(levels)) levels <- as.character(levels)
  if (!is.atomic(levels) || !is.null(dim(levels)) || anyNA(levels) ||
      length(levels) < 2L || length(levels) > 1024L) {
    stop("target_levels must be an ordered vector of 2 to 1024 public values.",
         call. = FALSE)
  }
  if (is.character(levels)) {
    levels <- enc2utf8(levels)
    if (any(!nzchar(levels))) {
      stop("target_levels cannot contain empty strings.", call. = FALSE)
    }
  } else if (is.numeric(levels)) {
    levels <- as.numeric(levels)
    if (any(!is.finite(levels))) {
      stop("Numeric target_levels must be finite.", call. = FALSE)
    }
  } else if (!is.logical(levels)) {
    stop("target_levels must be character, logical, or numeric.", call. = FALSE)
  }
  if (anyDuplicated(levels)) {
    stop("target_levels must be unique and ordered.", call. = FALSE)
  }
  n_classes <- suppressWarnings(as.integer(n_classes))
  if (length(n_classes) != 1L || is.na(n_classes) ||
      length(levels) != n_classes) {
    stop("target_levels length must equal the model's public class count.",
         call. = FALSE)
  }
  list(levels = levels, bounds = NULL)
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
  base <- c("flwr[app]>=1.31.0,<1.32.0", "numpy>=1.21.0", "pandas>=1.3.0",
            "pyarrow>=10.0.0", "torch>=2.0.0,<3.0.0", "opacus>=1.4.0,<2.0.0",
            "cryptography>=42.0.0")
  if (!isTRUE(vision)) return(base)
  c(base, "torchvision>=0.15.0,<1.0.0", "pillow>=9.0.0",
    "nibabel>=5.0.0", "pynrrd>=1.0.0", "SimpleITK>=2.2.0", "monai>=1.3.0")
}

# Legacy internal helper retained for source compatibility only. It is no longer
# called by the submission pipeline: hardened nodes disable exact feature
# statistics, and new runs accept only data-independent public bounds through
# `feature_bounds`.
.compute_feature_norm <- function(conns, symbol, features) {
  features <- as.character(features)
  if (!length(features)) return(NULL)
  expr <- call("flowerFeatureStatsDS", symbol, .ds_encode(features))
  res <- tryCatch(DSI::datashield.aggregate(conns, expr), error = function(e) NULL)
  if (is.null(res) || !length(res)) return(NULL)

  z <- stats::setNames(numeric(length(features)), features)
  tot_n <- z; tot_sum <- z; tot_sumsq <- z
  any_data <- FALSE
  for (st in res) {
    if (is.null(st) || is.null(st$features)) next
    fn <- as.character(unlist(st$features))
    nn <- as.numeric(unlist(st$n));     ss <- as.numeric(unlist(st$sum))
    sq <- as.numeric(unlist(st$sumsq))
    for (i in seq_along(fn)) {
      f <- fn[i]
      if (!(f %in% features)) next
      if (length(nn) >= i && is.finite(nn[i])) { tot_n[f]    <- tot_n[f]    + nn[i]; any_data <- TRUE }
      if (length(ss) >= i && is.finite(ss[i]))   tot_sum[f]  <- tot_sum[f]  + ss[i]
      if (length(sq) >= i && is.finite(sq[i]))   tot_sumsq[f]<- tot_sumsq[f]+ sq[i]
    }
  }
  if (!any_data) return(NULL)

  means <- numeric(length(features)); sds <- numeric(length(features))
  for (i in seq_along(features)) {
    ni <- tot_n[i]
    if (is.na(ni) || ni < 1) { means[i] <- 0; sds[i] <- 1; next }   # absent feature -> no-op
    mu <- tot_sum[i] / ni
    v  <- tot_sumsq[i] / ni - mu * mu
    if (!is.finite(v) || v < 0) v <- 0
    sdv <- sqrt(v)
    if (!is.finite(mu)) mu <- 0
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1   # constant/degenerate feature -> no scaling
    means[i] <- mu; sds[i] <- sdv
  }
  list(means = means, sds = sds)
}
