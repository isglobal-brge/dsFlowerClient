# Module: Unified submission (researcher side)
#
# The single researcher-facing path that turns a model spec into a running
# federated DP job. There is NO server-side catalog: the client codegens the
# submission (a declarative neural graph), ships it in ONE FAB
# (ServerApp + ClientApp + data-only model spec), the nodes spool it for the run, enforce DP
# node-side (DP-SGD / egress gate per the manifest dp-track), and delete
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

#' Validate one exact node capability response against the bundled runner
#' @keywords internal
.validate_runner_compatibility <- function(raw_caps, conns, expected_hash) {
  caps <- .dsi_exact_node_results(raw_caps, conns)
  missing <- if (is.null(caps)) names(conns) else
    names(conns)[vapply(caps, is.null, logical(1))]
  if (is.null(caps) || length(missing)) {
    detail <- if (length(missing)) paste0(" on: ", paste(missing, collapse = ", ")) else ""
    stop("Could not verify the dsFlower runner compatibility: no node capabilities returned",
         detail, ".",
         call. = FALSE)
  }

  node_names <- names(caps)
  failures <- character()
  for (i in seq_along(caps)) {
    cap <- caps[[i]]
    abi <- if (is.list(cap)) {
      tryCatch(suppressWarnings(as.numeric(cap[["runner_abi"]])),
               error = function(e) numeric())
    } else numeric()
    remote_hash <- if (is.list(cap)) {
      tryCatch(tolower(as.character(cap[["runner_sha256"]])),
               error = function(e) character())
    } else character()
    abi_ok <- length(abi) == 1L && !is.na(abi) && is.finite(abi) &&
      abi == floor(abi) && abi == 3
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
         "; expected runner_abi=3 and runner_sha256=",
         substr(expected_hash, 1L, 12L),
         ". Update dsFlower/dsFlowerClient so their bundled runners match.",
         call. = FALSE)
  }
  invisible(caps)
}

#' Fail early when a node cannot execute this client's trusted runner
#' @keywords internal
.assert_runner_compatibility <- function(conns) {
  expected_hash <- .compute_local_runner_hash()
  raw_caps <- tryCatch(
    DSI::datashield.aggregate(
      conns, expr = call("flowerGetCapabilitiesDS", "none")),
    error = function(e) {
      stop("Could not verify the dsFlower runner compatibility: ",
           conditionMessage(e), call. = FALSE)
    }
  )
  .validate_runner_compatibility(raw_caps, conns, expected_hash)
}

#' Require a fresh operational probe for one native-tree engine on every node
#' @keywords internal
.assert_native_tree_capability <- function(conns, engine) {
  if (!is.character(engine) || length(engine) != 1L || is.na(engine) ||
      !engine %in% .NATIVE_TREE_ENGINES) {
    stop("Native-tree engine has no implemented capability probe.",
         call. = FALSE)
  }
  expected_hash <- .compute_local_runner_hash()
  raw_caps <- tryCatch(
    DSI::datashield.aggregate(
      conns, expr = call("flowerGetCapabilitiesDS", engine)),
    error = function(e) {
      stop("Could not verify native-tree capability for '", engine, "': ",
           conditionMessage(e), call. = FALSE)
    })
  caps <- .validate_runner_compatibility(raw_caps, conns, expected_hash)
  unavailable <- character()
  for (i in seq_along(caps)) {
    native <- if (is.list(caps[[i]])) caps[[i]][["native_tree"]] else NULL
    availability <- paste0(engine, "_native_tight_available")
    probed <- if (is.list(native)) {
      as.character(unlist(native[["probed_engines"]], use.names = FALSE))
    } else {
      character()
    }
    ready <- is.list(native) && identical(probed, engine) &&
      identical(as.character(native[["contract"]] %||% ""),
                .NATIVE_TREE_CONTRACT) &&
      isTRUE(native[[availability]])
    if (!ready) unavailable <- c(unavailable, names(caps)[[i]])
  }
  if (length(unavailable)) {
    detail <- if (length(unavailable)) {
      paste0(" on: ", paste(unavailable, collapse = ", "))
    } else {
      ""
    }
    stop("Verified native-tree capability for '", engine,
         "' is unavailable", detail, ".",
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

  native_tree <- identical(sub$track %||% NULL, "native_tree")
  native_validation <- identical(
    sub$track %||% NULL, "native_tree_validation")
  association <- identical(sub$track %||% NULL, "association")
  dependency_light_runtime <- native_tree || native_validation || association
  server_ref <- if (association) {
    "dsflower_runner.association_server_app:app"
  } else if (native_tree) {
    "dsflower_runner.native_tree_server_app:app"
  } else if (native_validation) {
    "dsflower_runner.native_tree_validation_server_app:app"
  } else {
    "dsflower_runner.server_app:app"
  }
  client_ref <- if (association) {
    "dsflower_runner.association_client_app:app"
  } else if (native_tree) {
    "dsflower_runner.native_tree_client_app:app"
  } else if (native_validation) {
    "dsflower_runner.native_tree_validation_client_app:app"
  } else {
    "dsflower_runner.client_app:app"
  }
  dependencies <- if (dependency_light_runtime) {
    .native_tree_dependencies()
  } else {
    .harness_dependencies(vision)
  }
  toml <- paste0(
    '[build-system]\nrequires = ["hatchling"]\n',
    'build-backend = "hatchling.build"\n\n',
    '[project]\nname = "dsflower-app"\nversion = "1.0.0"\n',
    'description = "dsFlower submission"\nlicense = "MIT"\n',
    'dependencies = [', paste0('"', dependencies, '"', collapse = ", "), ']\n\n',
    '[tool.hatch.build.targets.wheel]\npackages = ', pkg_toml, '\n\n',
    '[tool.flwr.app]\npublisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "', server_ref, '"\n',
    'clientapp = "', client_ref, '"\n\n',
    '[tool.flwr.app.config]\n', paste(config_lines, collapse = "\n"), '\n'
  )
  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  app_dir
}

.neural_training_config <- function(params, loss_name) {
  p <- params %||% list()
  optimizer <- as.character(p[["optimizer"]] %||% "sgd")
  scheduler <- as.character(p[["scheduler"]] %||% "none")
  out <- list(
    "learning-rate" = as.numeric(p[["learning_rate"]] %||% 0.01),
    "weight-decay" = as.numeric(p[["weight_decay"]] %||% 0),
    "l1-penalty" = as.numeric(p[["l1_penalty"]] %||% 0),
    "optimizer-name" = optimizer,
    "scheduler-name" = scheduler)
  if (identical(loss_name, "negbin_nll")) {
    out[["nb-dispersion"]] <- as.numeric(p[["nb_dispersion"]] %||% 1)
  } else if (identical(loss_name, "gamma_nll")) {
    out[["gamma-shape"]] <- as.numeric(p[["gamma_shape"]] %||% 1)
  } else if (identical(loss_name, "huber")) {
    out[["huber-delta"]] <- as.numeric(p[["huber_delta"]] %||% 1)
  } else if (identical(loss_name, "quantile")) {
    out[["quantile-level"]] <- as.numeric(p[["quantile"]] %||% 0.5)
  }
  if (identical(optimizer, "sgd")) {
    out <- c(out, list(
      "optimizer-momentum" = as.numeric(p[["momentum"]] %||% 0),
      "optimizer-nesterov" = isTRUE(p[["nesterov"]])))
  } else if (optimizer %in% c("adam", "adamw")) {
    out <- c(out, list(
      "optimizer-beta1" = as.numeric(p[["beta1"]] %||% 0.9),
      "optimizer-beta2" = as.numeric(p[["beta2"]] %||% 0.999),
      "optimizer-eps" = as.numeric(p[["optimizer_eps"]] %||% 1e-8),
      "optimizer-amsgrad" = isTRUE(p[["amsgrad"]])))
  } else if (identical(optimizer, "rmsprop")) {
    out <- c(out, list(
      "optimizer-momentum" = as.numeric(p[["momentum"]] %||% 0),
      "optimizer-eps" = as.numeric(p[["optimizer_eps"]] %||% 1e-8),
      "optimizer-rmsprop-alpha" = as.numeric(p[["rmsprop_alpha"]] %||% 0.99)))
  }
  if (identical(scheduler, "step")) {
    out <- c(out, list(
      "scheduler-step-size" = as.integer(p[["scheduler_step_size"]] %||% 1L),
      "scheduler-gamma" = as.numeric(p[["scheduler_gamma"]] %||% 0.1)))
  } else if (identical(scheduler, "exponential")) {
    out[["scheduler-gamma"]] <- as.numeric(p[["scheduler_gamma"]] %||% 0.1)
  } else if (identical(scheduler, "cosine")) {
    out[["scheduler-min-lr"]] <- as.numeric(p[["scheduler_min_lr"]] %||% 0)
  }
  out
}

.validate_scheduler_horizon <- function(sub, num_rounds) {
  if (!identical(sub$track, "neural")) return(invisible(TRUE))
  p <- sub$params %||% list()
  scheduler <- as.character(p[["scheduler"]] %||% "none")
  if (identical(scheduler, "none")) return(invisible(TRUE))
  total_epochs <- as.numeric(num_rounds) *
    as.numeric(p[["local_epochs"]] %||% 1L)
  if (!is.finite(total_epochs) || total_epochs < 2) {
    stop("A learning-rate scheduler requires at least two total local epochs ",
         "across all federated rounds.", call. = FALSE)
  }
  learning_rate <- as.numeric(p[["learning_rate"]] %||% 0.01)
  if (scheduler %in% c("step", "exponential")) {
    gamma <- as.numeric(p[["scheduler_gamma"]] %||% 0.1)
    if (identical(gamma, 1)) {
      stop("scheduler_gamma = 1 would leave the learning rate unchanged.",
           call. = FALSE)
    }
    exponent <- total_epochs - 1
    if (identical(scheduler, "step")) {
      step_size <- as.numeric(p[["scheduler_step_size"]] %||% 1L)
      if (step_size >= total_epochs) {
        stop("scheduler_step_size must be smaller than the total local-epoch ",
             "horizon.", call. = FALSE)
      }
      exponent <- floor((total_epochs - 1) / step_size)
    }
    if (gamma > 1 &&
        log(learning_rate) + exponent * log(gamma) > log(10) + 1e-12) {
      stop("The requested scheduler would raise learning_rate above 10 within ",
           "the training horizon.", call. = FALSE)
    }
  } else if (identical(scheduler, "cosine") &&
             identical(as.numeric(p[["scheduler_min_lr"]] %||% 0), learning_rate)) {
    stop("scheduler_min_lr equal to learning_rate would make the cosine ",
         "scheduler ineffective.", call. = FALSE)
  }
  invisible(TRUE)
}

.model_spec_preflight_cache <- new.env(parent = emptyenv())

.validate_declarative_model_preflight <- function(sub, features, data_kind) {
  if (!identical(sub$track, "neural")) {
    return(invisible(TRUE))
  }
  if (!identical(data_kind, "image") &&
      (is.null(features) || !length(features))) {
    stop("Declarative tabular models require a public feature geometry.",
         call. = FALSE)
  }
  params <- sub$params %||% list()
  input_dim <- if (identical(data_kind, "image")) {
    switch(as.character(params[["backbone"]] %||% "resnet18"),
      resnet18 = 512L, resnet18_3d = 512L,
      densenet121 = 1024L, densenet121_3d = 1024L,
      stop("Unsupported trusted image backbone.", call. = FALSE))
  } else {
    as.integer(length(features))
  }
  payload <- list(
    spec = sub$spec,
    loss_name = sub$loss,
    input_dim = input_dim,
    num_classes = as.integer(params[["n_classes"]] %||% 2L),
    num_labels = as.integer(params[["num_labels"]] %||% 2L))
  cache_key <- digest::digest(payload, algo = "sha256")
  if (exists(cache_key, envir = .model_spec_preflight_cache,
             inherits = FALSE)) {
    return(invisible(TRUE))
  }

  .ensure_client_framework("pytorch")
  script <- system.file(
    "python", "validate_model_spec.py", package = "dsFlowerClient")
  if (!nzchar(script)) {
    script <- file.path("inst", "python", "validate_model_spec.py")
  }
  if (!file.exists(script)) {
    stop("Bundled declarative model validator not found.", call. = FALSE)
  }
  contract <- tempfile(fileext = ".json")
  on.exit(unlink(contract), add = TRUE)
  jsonlite::write_json(
    payload, contract, auto_unbox = TRUE, null = "null", digits = NA)
  result <- processx::run(
    command = .client_python_cmd(), args = c(script, contract),
    env = .client_venv_env(), error_on_status = FALSE, timeout = 60)
  if (result$status != 0L) {
    detail <- trimws(result$stderr %||% "")
    if (!nzchar(detail)) detail <- "the trusted model geometry rejected it"
    stop("Declarative model preflight failed: ", detail, call. = FALSE)
  }
  assign(cache_key, TRUE, envir = .model_spec_preflight_cache)
  invisible(TRUE)
}

#' Submit + run a federated DP job from a model spec (the "pack" API)
#'
#' Codegens the submission, ships it in one FAB to the DataSHIELD SuperNodes,
#' runs it under the manifest-pinned enforced-DP track, and tears it down.
#'
#' @param conns DSI connections.
#' @param model A model name or \code{dsflower_model} (registry-resolved).
#' @param target Character; one target column, or exactly \code{num_labels}
#'   distinct columns for a multilabel model.
#' @param features Character vector; feature columns.
#' @param data Optional character data source resolved during connection.
#' @param resource Optional Opal resource name.
#' @param symbol Character; server-side data handle symbol.
#' @param num_rounds Integer; federated rounds.
#' @param feature_bounds Optional public feature bounds as
#'   \code{list(lower=..., upper=...)} in feature order. These constants are
#'   supplied without querying node data and define a clipped affine transform.
#' @param feature_cuts Required for native-tight tree models: one strictly
#'   increasing public cut vector per feature, inside \code{feature_bounds}.
#'   Seven data-independent cuts per feature are a practical
#'   benchmark-backed starting point; they are never inferred from private data.
#' @param target_levels Optional ordered public label vocabulary for
#'   classification. Non-numeric targets require it; node values are never used
#'   to infer label codes, and missing or unknown values map to public code zero.
#'   Multilabel applies one public two-level vocabulary to each target
#'   independently. Binary native-tree models require exactly two ordered levels so the
#'   saved validation contract retains identical label semantics.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count targets. The node clips each target to these constants.
#' @param allow_insecure_http Character vector of exact connection names allowed
#'   to use plaintext HTTP. Empty by default. This exception does not provide
#'   transport security; use it only behind an independently trusted network.
#' @param model_params Named list; params merged over the model's defaults.
#'   Neural learning rates must be finite and in \code{(0, 10]}.
#' @param data_kind "tabular" or "image".
#' @param strategy Character strategy name or a \code{dsflower_strategy} object.
#' @param output_dir Optional model output directory.
#' @param output_name Optional model output name.
#' @param torch_backend Character; node torch backend selection.
#' @param verbose Logical.
#' @param silent Logical; suppress progress feedback.
#' @param holdout Optional numeric test fraction for atomic tabular neural
#'   holdout validation. Unsupported tracks fail before private preparation.
#' @param cross_validation Optional integer in \code{[2, 10]} selecting a
#'   dedicated federated cross-validation job. Prefer the user-facing
#'   \code{ds.flower.cross_validate()} wrapper.
#' @return A \code{dsflower_run}, or a \code{dsflower_cv} when
#'   \code{cross_validation} is set.
#' @export
ds.flower.submit <- function(conns, model, target, features = NULL,
                             data = NULL, resource = NULL, symbol = NULL,
                             num_rounds = 1L,
                             model_params = list(), data_kind = "tabular",
                             strategy = "fedavg",
                             output_dir = NULL, output_name = NULL,
                             torch_backend = "auto", verbose = FALSE,
                             silent = FALSE, feature_bounds = NULL,
                             feature_cuts = NULL,
                             target_levels = NULL, target_bounds = NULL,
                             holdout = NULL,
                             cross_validation = NULL,
                             allow_insecure_http = getOption(
                               "dsflower.dsi_allow_insecure_http", character())) {
  holdout_spec <- .normalize_holdout(holdout)
  cv_spec <- .normalize_cross_validation(cross_validation)
  if (!is.null(holdout_spec) && !is.null(cv_spec)) {
    stop("'holdout' and 'cross_validation' cannot be combined.", call. = FALSE)
  }
  torch_backend <- .validate_torch_backend(torch_backend)
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
  if (!is.character(data_kind) || length(data_kind) != 1L || is.na(data_kind) ||
      !data_kind %in% c("tabular", "image")) {
    stop("data_kind must be exactly 'tabular' or 'image'.", call. = FALSE)
  }
  if (!is.null(features)) {
    if (!is.character(features) || !length(features) || anyNA(features) ||
        any(!nzchar(features)) || anyDuplicated(features)) {
      stop("'features' must contain unique, non-empty character column names.",
           call. = FALSE)
    }
    features <- enc2utf8(features)
  }
  if (identical(data_kind, "image") && !is.null(features)) {
    stop("Image training derives frozen-backbone features node-side; ",
         "do not pass tabular 'features'.", call. = FALSE)
  }
  if (identical(data_kind, "image") && !is.null(feature_bounds)) {
    stop("feature_bounds applies only to tabular features and cannot be used ",
         "with data_kind = 'image'.", call. = FALSE)
  }
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  registered_model <- .dsflower_get_model(model$name)
  if (!data_kind %in% (registered_model$data_kinds %||% "tabular")) {
    stop("Model '", model$name, "' does not support data_kind = '", data_kind,
         "'. Supported kind(s): ",
         paste(registered_model$data_kinds %||% "tabular", collapse = ", "),
         ".", call. = FALSE)
  }
  # Validate a directly supplied model object as well as call-level overrides.
  # This closes the low-level submit() path that previously bypassed the model
  # schema with a raw modifyList() merge.
  base_params <- .dsflower_resolve_model_params(
    registered_model, model$params %||% list())
  registered_model$defaults <- base_params
  model$params <- .dsflower_resolve_model_params(
    registered_model, model_params)
  sub <- .emit_submission(model)
  if (!is.null(holdout_spec)) {
    .assert_holdout_supported(sub, data_kind)
  }
  if (!is.null(cv_spec)) {
    .assert_cross_validation_supported(sub, data_kind)
  }
  target <- .validate_submission_target(sub, target)
  request <- NULL
  if (identical(sub$track, "native_tree")) {
    if (!identical(num_rounds, 1L)) {
      stop("Native-tree engines train their complete public schedule in exactly ",
           "one Flower round; set num_rounds = 1.", call. = FALSE)
    }
    if (!identical(data_kind, "tabular")) {
      stop("Native-tree engines accept data_kind = 'tabular' only.",
           call. = FALSE)
    }
    if (is.null(features) || !length(features)) {
      stop("Native-tree engines require an explicit ordered public feature list; ",
           "server schema auto-detection is not permitted.", call. = FALSE)
    }
    if (is.null(feature_bounds)) {
      stop("Native-tree engines require explicit public feature_bounds.",
           call. = FALSE)
    }
    if (is.null(feature_cuts)) {
      stop("Native-tree engines require complete public feature_cuts.",
           call. = FALSE)
    }
    public_bounds <- .validate_public_feature_bounds(feature_bounds, features)
    public_target <- .validate_public_target_spec(
      target_levels, target_bounds,
      task_type = if (identical(sub$task, "regression"))
        "regression" else "classification",
      loss_name = sub$loss, n_classes = 2L)
    if (identical(sub$task, "binary") && is.null(public_target$levels)) {
      stop("Binary native-tree training requires exactly two ordered public ",
           "target_levels ",
           "so its training and validation label semantics are identical.",
           call. = FALSE)
    }
    request <- .build_native_tree_request(
      sub$engine, sub$params, features,
      public_bounds[c("lower", "upper")], feature_cuts,
      target_name = target, target_levels = public_target$levels,
      target_bounds = public_target$bounds)
  } else if (!is.null(feature_cuts)) {
    stop("feature_cuts is only valid for native-tight tree models.",
         call. = FALSE)
  }
  .validate_scheduler_horizon(sub, num_rounds)
  lr <- suppressWarnings(as.numeric(
    (sub$params %||% list())[["learning_rate"]] %||% 0.01))
  if (length(lr) != 1L || is.na(lr) || !is.finite(lr) || lr <= 0 || lr > 10) {
    stop("learning_rate must be one finite value in (0, 10].", call. = FALSE)
  }

  # Aggregation strategy. All of these run ONLY on the researcher-side SuperLink,
  # over the already-DP client updates -> pure post-processing, so the (epsilon,
  # delta) guarantee is unchanged whichever is chosen (it is never a privacy knob).
  strategy_spec <- if (inherits(strategy, "dsflower_strategy")) {
    .canonicalize_strategy(strategy)
  } else {
    ds.flower.strategy(strategy)
  }
  local_learning_rate <- as.numeric(
    (sub$params %||% list())[["learning_rate"]] %||% 0.01)
  strategy_config <- .strategy_config_values(
    strategy_spec, client_learning_rate = local_learning_rate)
  strategy_lines <- .strategy_config_lines(
    strategy_spec, client_learning_rate = local_learning_rate)

  if (!is.null(features)) {
    early_features <- unique(setdiff(
      as.character(features), as.character(target)))
    .validate_submission_feature_geometry(
      model$name, sub, early_features)
  }

  # All validation above is data-independent. Verify the outer transport before
  # the first public capability DSI request, then stop before CLI/private staging
  # if any selected node lacks the trusted native runtime.
  preflight_capabilities <- NULL
  if (identical(sub$track, "native_tree")) {
    suppressWarnings(.validate_dsi_transport_security(
      conns, allow_insecure_http = allow_insecure_http))
    preflight_capabilities <-
      .assert_native_tree_capability(conns, sub$engine)
  }
  .require_flwr_cli()
  if (!is.null(features)) {
    .validate_declarative_model_preflight(
      sub,
      unique(setdiff(as.character(features), as.character(target))),
      data_kind)
  }
  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)

  if (is.null(data) && is.null(resource) && is.null(symbol)) symbol <- "D"

  # Reject an accidental transport downgrade before the first DSI request
  # (including schema auto-detection, handle creation, staging or app upload).
  # link.up validates again immediately before starting the tunnel and emits the
  # single warning for an explicitly allowed HTTP site.
  if (!identical(sub$track, "native_tree")) {
    suppressWarnings(.validate_dsi_transport_security(
      conns, allow_insecure_http = allow_insecure_http))
  }

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
  .validate_submission_feature_geometry(
    model$name, sub, features)
  .validate_declarative_model_preflight(sub, features, data_kind)

  # No exact statistics leave a data node.  Optional scaling/ranges are public
  # constants supplied by the analyst before the run; DP-SGD still clips every
  # per-sample gradient independently of this utility-only transform.
  public_bounds <- .validate_public_feature_bounds(feature_bounds, features)
  if (!is.null(public_bounds) && verbose) {
    message("  Preprocessing: using public bounds for ", length(features),
            " feature(s); no node statistics were queried.")
  }

  task_type <- if (identical(sub$track, "native_tree")) {
    if (identical(sub$task, "regression")) "regression" else "classification"
  } else switch(sub$loss %||% "bce_logits",
                mse = "regression", huber = "regression",
                quantile = "regression", poisson_nll = "count",
                negbin_nll = "count", gamma_nll = "regression",
                "classification")
  p <- sub[["params"]] %||% list()
  n_target_classes <- as.integer(
    p[["n_classes"]] %||% p[["num_classes"]] %||% 2L)
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
  capabilities <- if (is.null(preflight_capabilities)) {
    .assert_runner_compatibility(conns)
  } else {
    preflight_capabilities
  }
  # Tabular runs need a non-empty feature set. The symbol path auto-detects above; data=/
  # resource= inputs must pass `features` explicitly -- fail with a clear message rather than
  # the downstream "num-features must be set" from the node.
  if ((is.null(features) || length(features) == 0L) && !identical(data_kind, "image")) {
    stop("No feature columns: pass `features` explicitly (auto-detect is only available for ",
         "the `symbol=` data path).", call. = FALSE)
  }
  n_features <- if (identical(data_kind, "image")) {
    .vision_extractor_feature_dim(p[["backbone"]])
  } else if (!is.null(features)) {
    length(features)
  } else {
    0L
  }

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
  holdout_contract <- NULL
  cv_contract <- NULL
  cv_capabilities <- NULL
  cv_job_sha256 <- NULL
  if (!is.null(holdout_spec)) {
    privacy_unit <- .validation_common_privacy_unit(capabilities)
    holdout_contract <- .holdout_contract(holdout_spec, privacy_unit)
    prepare_config <- c(prepare_config, .holdout_config(holdout_contract))
  } else if (!is.null(cv_spec)) {
    cv_capabilities <- .cross_validation_common_capabilities(capabilities)
    cv_contract <- .cross_validation_contract(
      cv_spec, cv_capabilities$privacy_unit)
    prepare_config <- c(
      prepare_config, .cross_validation_config(cv_contract))
  }
  if (identical(sub$track, "neural")) {
    p <- sub[["params"]]
    training_config <- .neural_training_config(p, sub$loss)
    prepare_config <- c(prepare_config, list(
      "model-spec-b64" = .spec_to_b64(sub$spec),
      "loss-name" = sub$loss %||% "bce_logits",
      "num-classes" = as.integer(
        p[["n_classes"]] %||% p[["num_classes"]] %||% 2L),
      "num-labels" = as.integer(p[["num_labels"]] %||% 2L),
      "local-epochs" = as.integer(p[["local_epochs"]] %||% 1L),
      "batch-size" = as.integer(p[["batch_size"]] %||% 32L)),
      training_config)
    if (identical(data_kind, "image")) {
      prepare_config <- c(prepare_config, list(
        "backbone" = p[["backbone"]],
        "image-size" = as.integer(p[["image_size"]]),
        "vision-extractor-profile" = p[["vision_extractor_profile"]]))
    }
  } else if (identical(sub$track, "native_tree")) {
    prepare_config <- c(prepare_config, list(
      "native-tree-request-b64" = request$b64,
      "native-tree-request-sha256" = request$sha256))
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
  if (!is.null(cv_contract)) {
    prepare_config <- c(
      prepare_config, strategy_config,
      list("cv-n-nodes" = as.integer(n_clients)))
    cv_job_sha256 <- .cv_job_sha256(
      prepare_config, features, target,
      runner_abi = cv_capabilities$runner_abi,
      runner_sha256 = cv_capabilities$runner_sha256,
      privacy_policy_sha256 = cv_capabilities$privacy_policy_sha256,
      privacy_clipping_norm = cv_capabilities$privacy_clipping_norm)
    prepare_config[["cv-job-sha256"]] <- cv_job_sha256
  }
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = prepare_config)

  if (!is.null(up)) {
    pin <- .pin_user_module(conns, hsym, up)
    if (verbose) message("  Pinned: ",
                         paste(unique(unlist(lapply(pin, `[[`, "pinned"))), collapse = ", "), ".")
  }

  results_root <- file.path(tempdir(), "dsflower_results")
  dir.create(results_root, recursive = TRUE, showWarnings = FALSE)
  results_dir <- tempfile(
    pattern = paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_"),
    tmpdir = results_root)
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cfg <- c(
    .toml_kv("dp-track", sub$track),
    .toml_kv("data-kind", data_kind),
    .toml_kv("task-type", task_type),
    strategy_lines,
    paste0("num-features = ", as.integer(n_features)),
    paste0("num-server-rounds = ", as.integer(num_rounds)),
    paste0("min-train-nodes = ", as.integer(n_clients)),
    .toml_kv("results-dir", results_dir)
  )
  if (!is.null(holdout_contract)) {
    holdout_fields <- .holdout_config(holdout_contract)
    cfg <- c(cfg, unname(vapply(names(holdout_fields), function(key) {
      .toml_kv(key, holdout_fields[[key]])
    }, character(1))))
    if (!is.null(public_target$bounds)) {
      cfg <- c(
        cfg,
        .toml_kv("holdout-target-lower", public_target$bounds$lower),
        .toml_kv("holdout-target-upper", public_target$bounds$upper))
    }
  } else if (!is.null(cv_contract)) {
    cv_fields <- .cross_validation_config(cv_contract)
    cfg <- c(cfg, unname(vapply(names(cv_fields), function(key) {
      .toml_kv(key, cv_fields[[key]])
    }, character(1))))
    if (!is.null(public_target$bounds)) {
      cfg <- c(
        cfg,
        .toml_kv("cv-target-lower", public_target$bounds$lower),
        .toml_kv("cv-target-upper", public_target$bounds$upper))
    }
    cfg <- c(
      cfg,
      .toml_kv("cv-n-nodes", as.integer(n_clients)),
      .toml_kv("cv-job-sha256", cv_job_sha256))
  }
  if (identical(sub$track, "neural")) {
    p <- sub[["params"]]
    training_config <- .neural_training_config(p, sub$loss)
    cfg <- c(cfg,
      .toml_kv("model-spec-b64", .spec_to_b64(sub$spec)),
      .toml_kv("loss-name", sub$loss %||% "bce_logits"),
      paste0("num-classes = ", as.integer(
        p[["n_classes"]] %||% p[["num_classes"]] %||% 2L)),
      paste0("num-labels = ", as.integer(p[["num_labels"]] %||% 2L)),
      paste0("local-epochs = ", as.integer(p[["local_epochs"]] %||% 1L)),
      paste0("batch-size = ", as.integer(p[["batch_size"]] %||% 32L)),
      unname(vapply(names(training_config), function(key) {
        .toml_kv(key, training_config[[key]])
      }, character(1))))
    if (!is.null(public_bounds))
      cfg <- c(cfg, .toml_kv("feature-bounds-b64", .spec_to_b64(public_bounds)))
    if (identical(data_kind, "image")) {
      cfg <- c(cfg, .toml_kv(
        "backbone", as.character(p[["backbone"]] %||% "resnet18")),
        paste0("image-size = ", as.integer(p[["image_size"]] %||% 224L)),
        .toml_kv("vision-extractor-profile",
                 p[["vision_extractor_profile"]]))
    }
  } else if (identical(sub$track, "native_tree")) {
    cfg <- c(
      cfg,
      .toml_kv("native-tree-request-b64", request$b64),
      .toml_kv("native-tree-request-sha256", request$sha256))
  }

  app_dir <- .build_submission_app(sub, cfg, results_dir,
                                   vision = identical(data_kind, "image"))
  if (!identical(sub$track, "native_tree")) {
    .ensure_client_framework("pytorch")
  }

  # link.up starts the loopback-only insecure SuperLink after validating the
  # outer DSI transport, then starts the per-node tunnel forwarders. Do not start
  # the SuperLink separately: a default SSL SuperLink mismatches the insecure
  # inner gRPC the SuperNodes speak and the run hangs.
  ds.flower.link.up(conns, allow_insecure_http = allow_insecure_http)
  recipe <- structure(list(
    model = list(name = model$name,
                 framework = model$framework %||% "pytorch", track = sub$track,
                 engine = sub$engine %||% NULL, task = sub$task %||% NULL),
    # Persist the exact public contract used to build the trusted model.  The
    # native checkpoint contains only tensors; without this data-only spec the
    # client cannot reconstruct CNN/DAG/recurrent models faithfully for
    # prediction or private validation.
    model_spec = sub$spec,
    model_params = sub$params %||% list(),
    loss_name = sub$loss %||% NULL,
    data_kind = data_kind,
    strategy = list(name = strategy_spec$name, params = strategy_spec$params),
    num_rounds = as.integer(num_rounds),
    features = features,
    feature_lower = if (!is.null(public_bounds)) public_bounds$lower else NULL,
    feature_upper = if (!is.null(public_bounds)) public_bounds$upper else NULL,
    target_levels = public_target$levels,
    target_bounds = public_target$bounds,
    native_tree_request_b64 = if (!is.null(request)) request$b64 else NULL,
    native_tree_request_sha256 = if (!is.null(request)) request$sha256 else NULL,
    public_schema_sha256 = if (!is.null(request)) {
      request$value$public_schema$sha256
    } else NULL,
    holdout_contract = holdout_contract,
    holdout_validation_bins = if (!is.null(holdout_contract))
      .HOLDOUT_VALIDATION_BINS else NULL,
    cross_validation_contract = cv_contract,
    cross_validation_bins = if (!is.null(cv_contract))
      .CV_VALIDATION_BINS else NULL,
    cross_validation_strategy = if (!is.null(cv_contract))
      strategy_config else NULL,
    cross_validation_n_nodes = if (!is.null(cv_contract))
      as.integer(n_clients) else NULL,
    cross_validation_job_sha256 = cv_job_sha256), class = "dsflower_recipe")

  # Cleanup (tunnel/handle/app/connection) is guaranteed by the on.exit above, on both a
  # run error and normal completion.
  ds.flower.nodes.ensure(
    conns, hsym,
    torch_backend = if (identical(sub$track, "native_tree")) NULL else torch_backend)
  ds.flower.run.start(recipe, conns, app_dir = app_dir,
                      results_dir = results_dir, symbol = hsym,
                      output_dir = output_dir, output_name = output_name,
                      verbose = verbose, silent = silent)
}

#' Validate target-column shape against the node-owned loss
#' @keywords internal
.validate_submission_target <- function(sub, target) {
  if (!is.character(target)) {
    stop("'target' must contain target column names.", call. = FALSE)
  }
  target <- enc2utf8(target)
  if (!length(target) || anyNA(target) || any(!nzchar(target)) ||
      anyDuplicated(target)) {
    stop("'target' must contain unique, non-empty target column names.",
         call. = FALSE)
  }
  if (identical(sub$loss, "multilabel_bce")) {
    num_labels <- suppressWarnings(as.integer(
      (sub[["params"]] %||% list())[["num_labels"]]))
    if (length(num_labels) != 1L || is.na(num_labels) ||
        num_labels < 2L || num_labels > 1024L) {
      stop("Multilabel models require public num_labels in [2, 1024].",
           call. = FALSE)
    }
    if (length(target) != num_labels) {
      stop("Multilabel model num_labels=", num_labels, " requires exactly ",
           num_labels, " target columns.", call. = FALSE)
    }
  } else if (length(target) != 1L) {
    stop("The selected model requires exactly one target column.", call. = FALSE)
  }
  if (identical(sub$loss, "bce_logits")) {
    n_classes <- as.integer(
      (sub[["params"]] %||% list())[["n_classes"]] %||% 2L)
    if (length(n_classes) != 1L || is.na(n_classes) || n_classes != 2L) {
      stop("bce_logits is binary only; use cross_entropy for multiclass models.",
           call. = FALSE)
    }
  }
  target
}

#' Validate architecture geometry against the public feature schema
#' @keywords internal
.validate_submission_feature_geometry <- function(model_name, sub, features) {
  if (is.null(features) || !length(features)) return(invisible(TRUE))
  params <- sub[["params"]] %||% list()
  expected <- switch(model_name,
    pytorch_cnn = prod(as.numeric(params[["input_shape"]])),
    pytorch_tcn = prod(as.numeric(params[["input_shape"]])),
    pytorch_resnet = prod(as.numeric(params[["input_shape"]])),
    pytorch_transformer = as.numeric(params[["n_tokens"]]) *
      as.numeric(params[["d_model"]]),
    pytorch_lstm = as.numeric(params[["n_tokens"]]) *
      as.numeric(params[["n_features"]]),
    pytorch_gru = as.numeric(params[["n_tokens"]]) *
      as.numeric(params[["n_features"]]),
    NULL)
  if (!is.null(expected) &&
      (length(expected) != 1L || !is.finite(expected) ||
       expected != length(features))) {
    stop("Model '", model_name, "' expects ", expected,
         " public features from its architecture parameters, but received ",
         length(features), ".", call. = FALSE)
  }
  invisible(TRUE)
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
  if (!is.numeric(bounds$lower) || is.logical(bounds$lower) ||
      !is.numeric(bounds$upper) || is.logical(bounds$upper)) {
    stop("feature_bounds lower/upper must be numeric vectors.", call. = FALSE)
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
    if (!is.numeric(bounds$lower) || is.logical(bounds$lower) ||
        !is.numeric(bounds$upper) || is.logical(bounds$upper)) {
      stop("target_bounds lower/upper must be numeric scalars.", call. = FALSE)
    }
    lower <- as.numeric(bounds$lower)
    upper <- as.numeric(bounds$upper)
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
  base <- c("flwr==1.31.0", "numpy>=1.21.0", "pandas>=1.3.0",
            "pyarrow>=10.0.0", "torch>=2.0.0,<3.0.0", "opacus>=1.4.0,<2.0.0",
            "cryptography>=42.0.0")
  if (!isTRUE(vision)) return(base)
  c(base, "torchvision>=0.15.0,<1.0.0", "pillow>=9.0.0",
    "nibabel>=5.0.0", "pynrrd>=1.0.0", "SimpleITK>=2.2.0", "monai>=1.3.0")
}

.native_tree_dependencies <- function() {
  c("flwr==1.31.0", "numpy==2.4.6", "pandas==3.0.3",
    "pyarrow==23.0.1", "cryptography==46.0.7")
}
