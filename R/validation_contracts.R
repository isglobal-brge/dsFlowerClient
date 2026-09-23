# Shared fixed public validation and CV geometry.
.validationCvContractExtension <- function(run_config) {
  out <- list()
  if (identical(run_config[["task-type"]], "segmentation") ||
      identical(run_config[["validation-task"]], "segmentation")) {
    keys <- c("backbone", "image-size", "vision-extractor-profile",
      "image_asset", "image_path_col", "mask_asset", "mask_path_col",
      "sample_id_col", "mask_empty_col", "mask-vocabulary",
      "segmentation-alpha", "segmentation-smooth", "segmentation-selection",
      "segmentation-checkpoint-sha256", "segmentation-output-shape",
      "segmentation-preprocessing")
    out$segmentation <- run_config[intersect(keys, names(run_config))]
    for (key in intersect(c("segmentation-alpha", "segmentation-smooth"),
                          names(out$segmentation))) {
      out$segmentation[[key]] <- as.numeric(out$segmentation[[key]])
    }
    if (!is.null(out$segmentation[["image-size"]])) {
      out$segmentation[["image-size"]] <- as.integer(out$segmentation[["image-size"]])
    }
  } else if (!is.null(run_config[["cv-contract-sha256"]]) &&
             !is.null(run_config[["backbone"]])) {
    out$vision <- run_config[c("backbone", "image-size", "vision-extractor-profile")]
  }
  if (!is.null(run_config[["survival-config-b64"]])) {
    out$survival <- run_config[intersect(c("survival-config-b64",
      "validation-survival-horizons", "validation-survival-nll-bound"), names(run_config))]
  }
  if (!is.null(run_config[["public-initialisation-manifest-sha256"]])) {
    out$public_initialisation <- run_config[c("public-initialisation-origin",
      "public-initialisation-manifest-sha256", "public-initialisation-checkpoint-sha256",
      "public-initialisation-encoder-sha256", "public-initialisation-identity-version")]
  }
  out
}

