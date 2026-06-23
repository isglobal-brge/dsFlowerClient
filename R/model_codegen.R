# Module: Client-side model codegen
#
# Turns a registered dsflower_model into the artifact shipped in the FAB:
#   * neural -> a tiny model package whose __init__.py defines
#       build_model(cfg) -> torch.nn.Module        (architecture ONLY)
#   * trees  -> a validated XGBoost SPEC (a data dict; never code)
# There is NO server-side catalog: this is the only place a model name becomes a
# concrete submission, entirely client-side.

#' Random lowercase+digit token (for unique temp package dirs).
#' @keywords internal
.dsflower_token <- function(n = 10L) {
  paste0("m", paste(sample(c(letters, 0:9), n, replace = TRUE), collapse = ""))
}

#' Resolve + emit a submission artifact from a model spec.
#'
#' @param model A \code{dsflower_model} object, or a model name (resolved via the
#'   registry). Extra \code{params} are merged over the registered defaults.
#' @return For neural: list(track="neural", pkg_dir, package, module, loss, params).
#'   For trees: list(track="trees", spec, params).
#' @keywords internal
.emit_submission <- function(model) {
  if (!inherits(model, "dsflower_model")) model <- ds.flower.model(model)
  m <- .dsflower_get_model(model$name)
  params <- utils::modifyList(m$defaults %||% list(), model$params %||% list())

  if (identical(m$track, "neural")) {
    src <- m$generate(params)
    if (!is.character(src) || length(src) != 1L || !nzchar(src)) {
      stop("neural model generator for '", model$name,
           "' must return python source (a single string).", call. = FALSE)
    }
    base <- file.path(tempdir(), paste0("dsflower_sub_", .dsflower_token()))
    pkg_dir <- file.path(base, "dsflower_user_model")
    if (dir.exists(base)) unlink(base, recursive = TRUE)
    dir.create(pkg_dir, recursive = TRUE, showWarnings = FALSE)
    # build_model lives in the package __init__, so the import module IS the
    # package basename -- which the node independently verifies against the single
    # uploaded package, closing the model-module spoof.
    writeLines(src, file.path(pkg_dir, "__init__.py"))
    list(track = "neural", pkg_dir = pkg_dir, package = "dsflower_user_model",
         module = "dsflower_user_model", loss = m$loss, params = params)

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
