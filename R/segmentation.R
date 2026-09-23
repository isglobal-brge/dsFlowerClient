# Public binary segmentation pins and local artifact reconstruction.
.SEGMENTATION_CHECKPOINT_SHA256 <- "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

.segmentation_decoder_init <- function(value = "random") {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !(identical(value, "random") ||
        grepl("\\Apublic:[a-z0-9][a-z0-9._-]{0,63}\\z", value, perl = TRUE))) {
    stop("decoder_init must be 'random' or 'public:<checkpoint-id>' with a ",
         "1-64 character lowercase ASCII checkpoint id.", call. = FALSE)
  }
  value
}

.segmentation_public_config <- function(params) {
  decoder_init <- .segmentation_decoder_init(params$decoder_init %||% "random")
  config <- list(
    "segmentation-checkpoint-sha256" = params$segmentation_checkpoint_sha256,
    "segmentation-selection" = params$segmentation_selection,
    "segmentation-preprocessing" = params$segmentation_preprocessing,
    "segmentation-output-shape" = params$segmentation_output_shape,
    "mask-vocabulary" = params$mask_values)
  if (!identical(decoder_init, "random")) {
    config[["segmentation-decoder-init"]] <- decoder_init
  }
  config
}

.segmentation_server_initialization <- function(prepared, params, node_names) {
  init <- .segmentation_decoder_init(params$decoder_init %||% "random")
  if (identical(init, "random")) return(NULL)
  sites <- prepared$per_site
  if (!is.list(sites) || !length(sites) || anyDuplicated(names(sites)) ||
      !setequal(names(sites), node_names)) {
    stop("Public decoder initialization requires every node's verified checkpoint.",
         call. = FALSE)
  }
  payloads <- lapply(sites, function(site) {
    if (!is.list(site) || !is.list(site$segmentation_public_initialization)) {
      stop("Node public decoder initialization payload is invalid.", call. = FALSE)
    }
    payload <- site$segmentation_public_initialization
    if (!is.list(payload$provenance) || !is.list(payload$provenance$manifest)) {
      stop("Node public decoder initialization payload is invalid.", call. = FALSE)
    }
    provenance <- payload$provenance
    manifest <- provenance$manifest
    checkpoint <- manifest$checkpoint
    scalar <- function(value) is.character(value) && length(value) == 1L && !is.na(value)
    sha256 <- function(value) scalar(value) && grepl("\\A[0-9a-f]{64}\\z", value, perl = TRUE)
    if (!is.list(payload) || !setequal(names(payload), c("provenance", "checkpoint_base64")) ||
        !is.list(provenance) || !is.list(manifest) || !is.list(checkpoint) ||
        !sha256(provenance$manifest_sha256) || !sha256(checkpoint$sha256) ||
        !identical(manifest$checkpoint_id, substring(init, 8L)) ||
        !identical(manifest$model_id, "pytorch_resnet18_segmentation") ||
        !identical(manifest$decoder, params$decoder) ||
        !is.numeric(checkpoint$size_bytes) || length(checkpoint$size_bytes) != 1L ||
        is.na(checkpoint$size_bytes) || checkpoint$size_bytes < 1 ||
        checkpoint$size_bytes > 2 * 1024^2 ||
        !scalar(payload$checkpoint_base64) ||
        nchar(payload$checkpoint_base64, type = "bytes") > 3 * 1024^2 ||
        nchar(payload$checkpoint_base64, type = "bytes") %% 4L != 0L ||
        !grepl("\\A[A-Za-z0-9+/]*={0,2}\\z", payload$checkpoint_base64, perl = TRUE)) {
      stop("Node public decoder initialization payload is invalid.", call. = FALSE)
    }
    bytes <- tryCatch(jsonlite::base64_dec(payload$checkpoint_base64),
                      error = function(e) raw())
    if (length(bytes) != checkpoint$size_bytes ||
        !identical(digest::digest(bytes, algo = "sha256", serialize = FALSE),
                   checkpoint$sha256)) {
      stop("Node public decoder checkpoint digest mismatch.", call. = FALSE)
    }
    encoded <- gsub("[\r\n]", "", jsonlite::base64_enc(charToRaw(enc2utf8(
      as.character(jsonlite::toJSON(payload, auto_unbox = TRUE,
                                    null = "null", digits = I(17)))))))
    if (nchar(encoded, type = "bytes") > 4 * 1024^2) {
      stop("Node public decoder initialization payload exceeds its byte limit.",
           call. = FALSE)
    }
    list(b64 = encoded, provenance = provenance)
  })
  first <- payloads[[1L]]
  if (!all(vapply(payloads, function(value) identical(value$b64, first$b64), logical(1)))) {
    stop("Nodes disagree on the public decoder checkpoint or its provenance.", call. = FALSE)
  }
  first
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
  .segmentation_decoder_init(p$decoder_init %||% "random")
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
