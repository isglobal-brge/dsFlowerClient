# Module: Client-side model codegen
#
# Turns a registered dsflower_model into the model SPEC (base64 JSON) shipped in
# the FAB and built with stock torch layers. Specs are DATA (never code); there
# is NO server-side catalog and NO researcher
# code in the FAB. This is the only place a model name becomes a concrete
# submission, entirely client-side.

.VISION_EXTRACTOR_PROFILES <- c(
  resnet18 = "dsflower-resnet18-imagenet1k-v1-extractor-v1",
  resnet18_3d = "dsflower-resnet18-monai-seed0-extractor-v1",
  densenet121 = "dsflower-densenet121-imagenet1k-v1-extractor-v1",
  densenet121_3d = "dsflower-densenet121-monai-seed0-extractor-v1")
.VISION_EXTRACTOR_FEATURE_DIMS <- c(
  resnet18 = 512L, resnet18_3d = 512L,
  densenet121 = 1024L, densenet121_3d = 1024L)
.VISION_EXTRACTOR_MIN_IMAGE_SIZES <- c(
  resnet18 = 1L, resnet18_3d = 1L,
  densenet121 = 29L, densenet121_3d = 128L)

.vision_extractor_profile <- function(backbone) {
  profile <- unname(.VISION_EXTRACTOR_PROFILES[[as.character(backbone)]])
  if (is.null(profile)) {
    stop("Unsupported trusted image backbone.", call. = FALSE)
  }
  profile
}

.vision_extractor_feature_dim <- function(backbone) {
  feature_dim <- unname(
    .VISION_EXTRACTOR_FEATURE_DIMS[[as.character(backbone)]])
  if (is.null(feature_dim)) {
    stop("Unsupported trusted image backbone.", call. = FALSE)
  }
  feature_dim
}

.vision_extractor_image_size <- function(backbone, image_size,
                                         field = "image_size") {
  minimum <- unname(
    .VISION_EXTRACTOR_MIN_IMAGE_SIZES[[as.character(backbone)]])
  if (is.null(minimum) || !is.numeric(image_size) || is.logical(image_size) ||
      length(image_size) != 1L || is.na(image_size) || !is.finite(image_size) ||
      image_size != floor(image_size) || image_size < minimum ||
      image_size > 512L) {
    stop(field, " is outside the canonical geometry for backbone '",
         as.character(backbone), "'.", call. = FALSE)
  }
  as.integer(image_size)
}

#' Resolve + emit a submission artifact from a model spec.
#'
#' @param model A \code{dsflower_model} object, or a model name (resolved via the
#'   registry). Extra \code{params} are merged over the registered defaults.
#' @return A list with track="neural", spec, loss, and params.
#' @keywords internal
.emit_submission <- function(model) {
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  m <- .dsflower_get_model(model$name)
  params <- utils::modifyList(m$defaults %||% list(), model$params %||% list())
  if (model$name %in% c("pytorch_resnet18", "pytorch_densenet121")) {
    params[["backbone"]] <- paste0(
      sub("^pytorch_", "", model$name),
      if (isTRUE(params[["volumetric"]])) "_3d" else "")
    params[["vision_extractor_profile"]] <-
      .vision_extractor_profile(params[["backbone"]])
    params[["image_size"]] <- .vision_extractor_image_size(
      params[["backbone"]], params[["image_size"]])
  }

  if (identical(m$track, "neural")) {
    spec <- m$generate(params)
    is_seq   <- !is.null(spec$layers)
    is_graph <- identical(spec$kind, "graph") && !is.null(spec$nodes)
    if (!is.list(spec) || !(is_seq || is_graph)) {
      stop("neural model generator for '", model$name,
           "' must return a model spec: a sequential list(kind = \"sequential\", ",
           "layers = ...) OR a typed-graph list(kind = \"graph\", nodes = ..., ",
           "output = ...).", call. = FALSE)
    }
    # The spec is DATA shipped in the run config (base64 JSON); there is NO user
    # package and NO researcher code in the FAB (pkg_dir absent), so the node builds
    # the architecture itself -- the import-time exfil surface simply does not exist.
    list(track = "neural", spec = spec, loss = m$loss, params = params)

  } else if (identical(m$track, "native_tree") &&
             m$engine %in% .NATIVE_TREE_ENGINES) {
    list(track = "native_tree", engine = m$engine, task = params$task,
         spec = NULL, loss = .dsflower_model_loss(m, params), params = params)
  } else {
    stop("model '", model$name, "' has unknown track '", m$track, "'.",
         call. = FALSE)
  }
}

#' Serialize a model spec to a base64 JSON string for transport as ONE TOML string
#' value. Base64 is ASCII, so the JSON (which contains quotes) needs no TOML
#' escaping; the node base64-decodes + json.loads it back into the spec it builds.
#' @keywords internal
.spec_to_b64 <- function(spec) {
  json <- jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null")
  enc <- jsonlite::base64_enc(charToRaw(enc2utf8(as.character(json))))
  gsub("[\r\n]", "", enc)            # one line: base64_enc MIME-wraps; a newline
}                                    # in the value would break the TOML config
