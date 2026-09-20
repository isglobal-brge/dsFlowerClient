# Public binary segmentation pins and local artifact reconstruction.
.SEGMENTATION_CHECKPOINT_SHA256 <- "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

.segmentation_public_config <- function(params) {
  list(
    "segmentation-checkpoint-sha256" = params$segmentation_checkpoint_sha256,
    "segmentation-selection" = params$segmentation_selection,
    "segmentation-preprocessing" = params$segmentation_preprocessing,
    "segmentation-output-shape" = params$segmentation_output_shape,
    "mask-vocabulary" = params$mask_values)
}

.resolve_segmentation_prediction_contract <- function(model_dir) {
  path <- file.path(model_dir, "metadata.json")
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$size) || isTRUE(info$isdir) ||
      info$size < 1 || info$size > .VALIDATION_METADATA_MAX_BYTES) {
    stop("Segmentation prediction requires bounded saved metadata.", call. = FALSE)
  }
  meta <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  p <- meta$model_params
  if (!identical(meta$model, "pytorch_resnet18_segmentation") ||
      !identical(meta$framework, "pytorch_vision") ||
      !identical(meta$track, "neural") ||
      !identical(meta$data_kind, "image") ||
      !identical(meta$loss_name, "segmentation_bce_dice") ||
      !identical(meta$privacy, "server-enforced-dp") ||
      !identical(meta$available, TRUE) || !identical(meta$status, "success") ||
      !is.list(p) || !is.list(meta$model_spec)) {
    stop("Segmentation prediction requires a successful saved DP release.",
         call. = FALSE)
  }
  expected <- .emit_submission(ds.flower.model.pytorch_resnet18_segmentation())$params
  pins <- c("backbone", "vision_extractor_profile", "image_size",
            "segmentation_checkpoint_sha256", "segmentation_selection",
            "segmentation_preprocessing", "segmentation_output_shape")
  for (key in pins) {
    if (!isTRUE(all.equal(p[[key]], expected[[key]], check.attributes = FALSE))) {
      stop("Saved segmentation profile pin is invalid: ", key, ".", call. = FALSE)
    }
  }
  decoder <- if (is.null(p$decoder)) "current" else p$decoder
  if (!is.character(decoder) || length(decoder) != 1L ||
      !decoder %in% c("current", "narrow", "pointwise") ||
      !is.numeric(p$alpha) || length(p$alpha) != 1L ||
      !p$alpha %in% c(0.5, 1) ||
      !is.character(p$mask_values) || length(p$mask_values) != 1L ||
      !p$mask_values %in% c("0,1", "0,255") ||
      !identical(.spec_to_b64(meta$model_spec),
                 .spec_to_b64(.segmentation_decoder_spec(decoder)))) {
    stop("Saved segmentation decoder or mask contract is invalid.", call. = FALSE)
  }
  artifact <- file.path(model_dir, "model.pt")
  info <- file.info(artifact)
  link <- Sys.readlink(artifact)
  if (!file.exists(artifact) || is.na(info$size) || isTRUE(info$isdir) ||
      info$size < 1 || info$size > .VISION_VALIDATION_ARTIFACT_MAX_BYTES ||
      (length(link) == 1L && !is.na(link) && nzchar(link))) {
    stop("Saved segmentation artifact must be a bounded regular file.", call. = FALSE)
  }
  list(artifact = normalizePath(artifact, winslash = "/", mustWork = TRUE),
       artifact_format = .VISION_VALIDATION_ARTIFACT_FORMAT,
       artifact_sha256 = digest::digest(file = artifact, algo = "sha256",
                                        serialize = FALSE),
       artifact_size_bytes = as.integer(info$size),
       track = "neural", task = "segmentation", data_kind = "image",
       model_spec = meta$model_spec, loss_name = meta$loss_name,
       n_classes = 2L, n_labels = 2L, feature_dim = 32768L,
       backbone = p$backbone, image_size = 128L,
       vision_extractor_profile = p$vision_extractor_profile,
       segmentation_config = c(.segmentation_public_config(p), list(
         "segmentation-alpha" = p$alpha, "segmentation-smooth" = 1)))
}

.format_segmentation_predictions <- function(value, expected_rows, type) {
  value <- tryCatch(as.matrix(value), error = function(e) NULL)
  if (is.null(value) || !is.numeric(value) ||
      !identical(dim(value), c(as.integer(expected_rows), 16384L)) ||
      any(!is.finite(value)) || any(value < 0 | value > 1) ||
      (identical(type, "response") && any(!value %in% c(0, 1)))) {
    stop("Local segmentation predictor returned invalid masks.", call. = FALSE)
  }
  # The wire is row-major pixels; expose [image, channel, row, column] in R.
  aperm(array(as.numeric(t(value)), dim = c(128L, 128L, 1L, expected_rows)),
        c(4L, 3L, 2L, 1L))
}

.assert_segmentation_hpo_supported <- function(loss) {
  if (identical(loss, "segmentation_bce_dice") &&
      isTRUE(.DSFLOWER_HPO_CONTEXT$active)) {
    stop("Private segmentation HPO is unsupported; HPO callbacks may only ",
         "score an already released segmentation model on local data.",
         call. = FALSE)
  }
  invisible(TRUE)
}
