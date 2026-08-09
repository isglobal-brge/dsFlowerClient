# Module: Client-side model codegen
#
# Turns a registered dsflower_model into the artifact shipped in the FAB:
#   * neural -> a model SPEC (base64 JSON) the node builds with stock torch layers
#   * trees  -> a validated dsFlower DP-GBDT spec (a data dict)
# Both are DATA (never code); there is NO server-side catalog and NO researcher
# code in the FAB. This is the only place a model name becomes a concrete
# submission, entirely client-side.

.canonicalize_gbdt_spec <- function(spec, model_name) {
  if (!is.list(spec) || is.null(names(spec)) ||
      any(!nzchar(names(spec))) || anyDuplicated(names(spec))) {
    stop("trees model generator for '", model_name,
         "' must return a uniquely named spec list.", call. = FALSE)
  }
  allowed <- c(
    "objective", "max_depth", "n_trees", "learning_rate", "reg_lambda",
    "n_bins", "feature_ranges", "target_bounds", "margin_bounds",
    "gradient_clip")
  unknown <- setdiff(names(spec), allowed)
  if (length(unknown)) {
    stop("Unknown DP-GBDT spec field(s) for '", model_name, "': ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  exact_integer <- function(value, name, default, lower, upper) {
    value <- value %||% default
    if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
        is.na(value) || !is.finite(value) || value != floor(value) ||
        value < lower || value > upper) {
      stop("DP-GBDT spec '", name, "' must be one integer in [",
           lower, ", ", upper, "].", call. = FALSE)
    }
    as.integer(value)
  }
  finite_number <- function(value, name, default, lower, upper) {
    value <- value %||% default
    if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
        is.na(value) || !is.finite(value) || value <= lower || value > upper) {
      stop("DP-GBDT spec '", name, "' must be in (", lower, ", ", upper,
           "].", call. = FALSE)
    }
    as.numeric(value)
  }
  interval <- function(value, name) {
    if (!is.numeric(value) || is.logical(value) || length(value) != 2L ||
        anyNA(value) || any(!is.finite(value)) || value[[1L]] >= value[[2L]] ||
        any(abs(value) > 1e6)) {
      stop("DP-GBDT spec '", name,
           "' must be a finite [lower, upper] interval.", call. = FALSE)
    }
    as.numeric(value)
  }
  objective <- spec[["objective"]] %||% "binary:logistic"
  if (!is.character(objective) || length(objective) != 1L ||
      is.na(objective) ||
      !objective %in% c("binary:logistic", "reg:squarederror")) {
    stop("DP-GBDT objective must be binary:logistic or reg:squarederror.",
         call. = FALSE)
  }
  out <- list(
    objective = objective,
    max_depth = exact_integer(spec[["max_depth"]], "max_depth", 3L, 1L, 6L),
    n_trees = exact_integer(spec[["n_trees"]], "n_trees", 50L, 1L, 200L),
    learning_rate = finite_number(
      spec[["learning_rate"]], "learning_rate", 0.3, 0, 10),
    reg_lambda = finite_number(
      spec[["reg_lambda"]], "reg_lambda", 1, 0, 1e6),
    n_bins = exact_integer(spec[["n_bins"]], "n_bins", 32L, 2L, 64L))

  ranges <- spec[["feature_ranges"]]
  if (!is.null(ranges)) {
    if (!is.list(ranges) || !length(ranges) || length(ranges) > 65536L) {
      stop("DP-GBDT feature_ranges must be a non-empty bounded interval list.",
           call. = FALSE)
    }
    out$feature_ranges <- lapply(
      seq_along(ranges), function(index) interval(
        ranges[[index]], paste0("feature_ranges[[", index, "]]")))
  }
  target_bounds <- spec[["target_bounds"]]
  margin_bounds <- spec[["margin_bounds"]]
  gradient_clip <- spec[["gradient_clip"]]
  if (identical(objective, "binary:logistic")) {
    if (!is.null(target_bounds) || !is.null(margin_bounds) ||
        !is.null(gradient_clip)) {
      stop("Regression bounds/gradient_clip cannot be used with binary:logistic.",
           call. = FALSE)
    }
  } else {
    if (!is.null(target_bounds)) {
      out$target_bounds <- interval(target_bounds, "target_bounds")
    }
    if (!is.null(margin_bounds)) {
      out$margin_bounds <- interval(margin_bounds, "margin_bounds")
    }
    if (!is.null(gradient_clip)) {
      out$gradient_clip <- finite_number(
        gradient_clip, "gradient_clip", NULL, 0, 2e6)
    }
  }
  out
}

#' Resolve + emit a submission artifact from a model spec.
#'
#' @param model A \code{dsflower_model} object, or a model name (resolved via the
#'   registry). Extra \code{params} are merged over the registered defaults.
#' @return For neural: list(track="neural", spec, loss, params).
#'   For trees: list(track="trees", spec, params).
#' @keywords internal
.emit_submission <- function(model) {
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  m <- .dsflower_get_model(model$name)
  params <- utils::modifyList(m$defaults %||% list(), model$params %||% list())
  if (model$name %in% c("pytorch_resnet18", "pytorch_densenet121")) {
    params[["backbone"]] <- paste0(
      sub("^pytorch_", "", model$name),
      if (isTRUE(params[["volumetric"]])) "_3d" else "")
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

  } else if (identical(m$track, "trees")) {
    spec <- .canonicalize_gbdt_spec(m$generate(params), model$name)
    list(track = "trees", spec = spec, params = params)

  } else {
    stop("model '", model$name, "' has unknown track '", m$track, "'.",
         call. = FALSE)
  }
}

#' Serialize a trees spec to the TOML config value (compact JSON string) the node
#' parses back into the validated DP-GBDT spec. The node clamps + pins it; this is
#' just transport of a data dict.
#' @keywords internal
.spec_to_toml_value <- function(spec) {
  jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null")
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
