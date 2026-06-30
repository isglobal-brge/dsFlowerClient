# Module: Client-side model codegen
#
# Turns a registered dsflower_model into the artifact shipped in the FAB:
#   * neural -> a model SPEC (base64 JSON) the node builds with stock torch layers
#   * trees  -> a validated XGBoost SPEC (a data dict)
# Both are DATA (never code); there is NO server-side catalog and NO researcher
# code in the FAB. This is the only place a model name becomes a concrete
# submission, entirely client-side.

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
    spec <- m$generate(params)
    if (!is.list(spec)) {
      stop("trees model generator for '", model$name,
           "' must return a spec list.", call. = FALSE)
    }
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
