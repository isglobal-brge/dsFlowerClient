# Analyst bundle validation uses the same closed verifier and admission as training.
.is_validation_checkpoint <- function(model) {
  is.character(model) && length(model) == 1L && !is.na(model) &&
    (startsWith(model, "client:") ||
      (dir.exists(path.expand(model)) &&
       file.exists(file.path(path.expand(model), "manifest.json"))) ||
      grepl("\\.zip$", model, ignore.case = TRUE))
}

.public_initialisation_identity <- function(summary, origin = "analyst-declared") {
  list("public-initialisation-origin" = origin,
    "public-initialisation-manifest-sha256" = summary$provenance$manifest_sha256,
    "public-initialisation-checkpoint-sha256" = summary$checkpoint_sha256,
    "public-initialisation-encoder-sha256" = summary$encoder_sha256,
    "public-initialisation-identity-version" = summary$identity_version)
}

.resolve_checkpoint_validation_contract <- function(model, bins) {
  path <- if (startsWith(model, "client:")) substring(model, 8L) else model
  if (!nzchar(path) || !file.exists(path.expand(path))) {
    stop("Validation checkpoint bundle does not exist.", call. = FALSE)
  }
  path <- normalizePath(path.expand(path), winslash = "/", mustWork = TRUE)
  .ensure_client_framework("pytorch")
  payload <- .segmentation_checkpoint_python("local", path, NULL)
  summary <- payload
  summary$local_arrays_b64 <- NULL
  summary <- .segmentation_public_summary(summary)
  manifest <- summary$provenance$manifest
  bins <- .validation_scalar_integer(bins, "bins", 4L, 512L)
  common <- list(model_dir = path, artifact = path, bins = bins, track = "neural",
    checkpoint = list(path = path, payload = payload, summary = summary),
    model_training_privacy = "analyst-declared-public")
  if (identical(manifest$role, "segmentation_decoder")) {
    model <- ds.flower.model.pytorch_resnet18_segmentation(
      decoder_init = "random", decoder = manifest$decoder)
    sub <- .emit_submission(model)
    p <- sub$params
    config <- .segmentation_public_config(p)
    config[["segmentation-alpha"]] <- p$alpha
    config[["segmentation-smooth"]] <- 1
    config[["image_asset"]] <- p$image_asset
    config[["mask_asset"]] <- p$mask_asset
    config[["image_path_col"]] <- p$image_path_col
    config[["sample_id_col"]] <- p$sample_id_col
    return(c(common, list(task = "segmentation", data_kind = "image",
      features = NULL, feature_bounds = NULL, target_bounds = NULL, target_levels = NULL,
      model_spec = sub$spec, loss_name = sub$loss, n_classes = 2L, n_labels = 2L,
      feature_dim = 32768L, backbone = p$backbone, image_size = 128L,
      vision_extractor_profile = p$vision_extractor_profile,
      segmentation_config = config)))
  }
  if (!identical(manifest$role, "tabular_model")) {
    stop("Checkpoint has no trusted private validation contract.", call. = FALSE)
  }
  cfg <- manifest$model_config
  schema <- manifest$feature_contract
  loss <- cfg[["loss-name"]]
  nc <- as.integer(cfg[["num-classes"]])
  task <- .validation_task_for_loss(loss, nc)
  survival <- if (.is_survival_loss(loss)) {
    .validate_survival_config(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
      cfg[["survival-config-b64"]])), simplifyVector = TRUE), loss)
  } else NULL
  bounds <- if (is.null(schema$feature_lower)) NULL else list(
    lower = as.numeric(unlist(schema$feature_lower, use.names = FALSE)),
    upper = as.numeric(unlist(schema$feature_upper, use.names = FALSE)))
  target_bounds <- if (is.null(schema$target_bounds)) NULL else list(
    lower = as.numeric(schema$target_bounds$lower), upper = as.numeric(schema$target_bounds$upper))
  if (task %in% c("regression", "count") && is.null(target_bounds)) {
    stop("Numeric checkpoint validation requires public target bounds.", call. = FALSE)
  }
  levels <- .validation_atomic(schema$target_levels)
  if (is.numeric(levels)) levels <- as.numeric(levels)
  target <- .validate_public_target_spec(levels, target_bounds,
    task_type = if (task %in% c("survival", "regression", "count")) task else "classification",
    loss_name = loss, n_classes = nc)
  levels <- target$levels
  target_bounds <- target$bounds
  c(common, list(task = task, data_kind = "tabular", features = as.character(
    unlist(schema$features, use.names = FALSE)), feature_bounds = bounds,
    target_bounds = target_bounds, target_levels = levels,
    model_spec = manifest$model_spec, loss_name = loss, n_classes = nc,
    n_labels = as.integer(cfg[["num-labels"]]), survival_config = survival,
    loss_config = .validation_loss_config(loss, cfg, wire = TRUE)))
}

.validation_task_for_loss <- function(loss, n_classes) {
  task <- switch(loss,
    bce_logits = "binary", cross_entropy = if (n_classes > 2L) "multiclass" else "binary",
    hinge = if (n_classes > 2L) "multiclass" else "binary", ordinal = "ordinal",
    multilabel_bce = "multilabel", mse = "regression", huber = "regression",
    quantile = "regression", gamma_nll = "regression", poisson_nll = "count",
    negbin_nll = "count", aft_weibull_nll = "survival", aft_lognormal_nll = "survival",
    discrete_hazard_nll = "survival", segmentation_bce_dice = "segmentation", NULL)
  if (is.null(task)) stop("Saved neural loss has no trusted validation semantics: ", loss, ".",
                          call. = FALSE)
  task
}

.private_survival_metric_config <- function(config, horizons = NULL, nll_bound = 20) {
  if (is.null(config)) {
    if (!is.null(horizons) || !identical(nll_bound, 20)) {
      stop("Survival metric options require a survival model.", call. = FALSE)
    }
    return(list())
  }
  horizons <- horizons %||% config$horizon
  if (!is.numeric(horizons) || is.logical(horizons) || !length(horizons) ||
      length(horizons) > 64L || anyNA(horizons) || any(!is.finite(horizons)) ||
      any(horizons < config$t_min) || any(horizons > config$horizon) || any(diff(horizons) <= 0)) {
    stop("survival_horizons must increase within the fitted public time domain.", call. = FALSE)
  }
  if (!is.numeric(nll_bound) || is.logical(nll_bound) || length(nll_bound) != 1L ||
      !is.finite(nll_bound) || nll_bound <= 0 || nll_bound > 1000) {
    stop("survival_nll_bound must be in (0, 1000].", call. = FALSE)
  }
  list("validation-survival-horizons" = as.character(jsonlite::toJSON(
    as.numeric(horizons), auto_unbox = FALSE, digits = I(17))),
    "validation-survival-nll-bound" = as.numeric(nll_bound))
}

.validation_loss_config <- function(loss, values, wire = FALSE) {
  key <- switch(loss, negbin_nll = "nb-dispersion", gamma_nll = "gamma-shape",
    huber = "huber-delta", quantile = "quantile-level", NULL)
  if (is.null(key)) return(list())
  source <- if (isTRUE(wire)) key else switch(key,
    `nb-dispersion` = "nb_dispersion", `gamma-shape` = "gamma_shape",
    `huber-delta` = "huber_delta", `quantile-level` = "quantile")
  value <- .validation_atomic(values[[source]]) %||%
    if (identical(loss, "quantile")) 0.5 else 1
  if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
      !is.finite(value) || value <= 0 ||
      (!identical(loss, "quantile") && value < 1e-6) ||
      value > (if (identical(loss, "huber")) 1e6 else 1e12) ||
      (identical(loss, "quantile") && value >= 1)) {
    stop("Saved public loss parameter is outside its trusted contract.", call. = FALSE)
  }
  setNames(list(as.numeric(value)), key)
}
