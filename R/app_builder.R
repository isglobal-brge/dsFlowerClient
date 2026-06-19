# Module: Flower App Generation (dsFlower 2.0)
# Builds the run's Flower app from the bundled, trusted Tier-1 harness skeleton.
# There are no server-side templates: the client copies the harness package and
# writes a pyproject.toml carrying only the model architecture + hyperparameters.
# All DP parameters come from the server-written manifest (node side), never from
# this client-controlled pyproject.

#' Build the Tier-1 harness Flower app for a run
#'
#' Copies the bundled \code{dsflower_harness} package into an app directory and
#' generates a \code{pyproject.toml} from the recipe. The same app is submitted
#' via \code{flwr run}; nodes run its ClientApp (always-on DP-SGD).
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param conns DSI connections (used only to size \code{min-train-nodes}).
#' @param app_dir Character; directory to create the app in (default: tempdir).
#' @param results_dir Character; directory for the ServerApp to save model/history.
#' @return Character; path to the created app directory.
#' @keywords internal
.build_flower_app <- function(recipe, conns = NULL, app_dir = NULL,
                              results_dir = NULL) {
  skeleton <- .harness_skeleton_dir()

  if (is.null(app_dir)) {
    app_dir <- file.path(tempdir(), "dsflower_app", "dsflower_harness")
  }
  if (dir.exists(app_dir)) unlink(app_dir, recursive = TRUE)
  dir.create(app_dir, recursive = TRUE, showWarnings = FALSE)

  # Copy the trusted harness package (dsflower_harness/) into the app dir.
  file.copy(skeleton, app_dir, recursive = TRUE)

  n_nodes <- if (is.null(conns)) 2L else max(1L, length(conns))
  .write_pyproject_toml(app_dir, recipe, results_dir = results_dir,
                        n_nodes = n_nodes)
  app_dir
}

#' Locate the bundled harness package skeleton (installed or source tree)
#' @keywords internal
.harness_skeleton_dir <- function() {
  p <- system.file("flower_app", "dsflower_harness", package = "dsFlowerClient")
  if (nzchar(p) && dir.exists(p)) return(p)
  # Dev fallback: source tree (devtools::load_all / non-installed).
  alt <- file.path("inst", "flower_app", "dsflower_harness")
  if (dir.exists(alt)) return(normalizePath(alt))
  stop("Bundled harness skeleton not found (flower_app/dsflower_harness).",
       call. = FALSE)
}

#' Map a recipe model to a harness architecture name (MVP: logreg, mlp)
#' @keywords internal
.harness_model <- function(recipe) {
  nm <- tolower(recipe$model$name %||% recipe$model$template %||% "")
  if (grepl("mlp", nm)) return("mlp")
  if (grepl("logreg|logistic|linear", nm)) return("logreg")
  stop("The Tier-1 harness currently supports logreg and mlp models (got '",
       nm, "'). Arbitrary models will be supported via Tier-2 app upload.",
       call. = FALSE)
}

#' Extract harness hyperparameters from the recipe (kebab-cased), with defaults
#' @keywords internal
.harness_hyperparams <- function(recipe) {
  p <- recipe$model$params %||% list()
  getp <- function(keys, default) {
    for (k in keys) if (!is.null(p[[k]])) return(p[[k]])
    default
  }
  list(
    `learning-rate` = as.numeric(getp(c("learning_rate", "lr"), 0.05)),
    `batch-size`    = as.integer(getp(c("batch_size"), 32L)),
    `local-epochs`  = as.integer(getp(c("local_epochs", "epochs"), 1L)),
    `hidden-dim`    = as.integer(getp(c("hidden_dim", "hidden"), 64L))
  )
}

#' Generate pyproject.toml for the harness app
#'
#' @param app_dir Character; path to the app directory.
#' @param recipe A \code{dsflower_recipe} object.
#' @param results_dir Character or NULL; ServerApp output directory.
#' @param n_nodes Integer; participating node count (sizes min-train-nodes).
#' @return Invisible NULL.
#' @keywords internal
.write_pyproject_toml <- function(app_dir, recipe, results_dir = NULL,
                                  n_nodes = 2L) {
  n_features <- length(recipe$features %||% character(0))
  if (n_features < 1L) {
    stop("The Tier-1 harness requires explicit feature columns so the global ",
         "model dimension is known (set `features = c(...)`).", call. = FALSE)
  }

  config_lines <- c(
    paste0("num-server-rounds = ", as.integer(recipe$num_rounds)),
    paste0("num-features = ", as.integer(n_features)),
    .toml_kv("model", .harness_model(recipe)),
    paste0("min-train-nodes = ", as.integer(min(n_nodes, max(1L, n_nodes))))
  )
  for (nm in names(hp <- .harness_hyperparams(recipe))) {
    config_lines <- c(config_lines, .toml_kv(nm, hp[[nm]]))
  }
  if (!is.null(results_dir)) {
    config_lines <- c(config_lines, .toml_kv("results-dir", results_dir))
  }

  toml <- paste0(
    '[build-system]\n',
    'requires = ["hatchling"]\n',
    'build-backend = "hatchling.build"\n\n',
    '[project]\n',
    'name = "dsflower_harness"\n',
    'version = "1.0.0"\n',
    'dependencies = [', paste0('"', .harness_dependencies(), '"',
                               collapse = ", "), ']\n\n',
    '[tool.hatch.build.targets.wheel]\n',
    'packages = ["dsflower_harness"]\n\n',
    '[tool.flwr.app]\n',
    'publisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "dsflower_harness.server_app:app"\n',
    'clientapp = "dsflower_harness.client_app:app"\n\n',
    '[tool.flwr.app.config]\n',
    paste(config_lines, collapse = "\n"), '\n'
  )
  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  invisible(NULL)
}

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

#' Pip dependencies for the harness app (fixed; one trusted app, not per-model)
#' @keywords internal
.harness_dependencies <- function() {
  c("flwr[app]>=1.27.0", "numpy>=1.21.0", "pandas>=1.3.0",
    "torch>=2.0.0", "opacus>=1.4.0")
}
