# Module: Flower App Generation
# Builds Flower App from composable recipe specs using templates.

#' Build a Flower App from a recipe
#'
#' Copies the appropriate template from \code{inst/flower_templates/} and
#' generates a \code{pyproject.toml} with run_config from the recipe.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param app_dir Character; directory to create the app in (default: tempdir).
#' @param results_dir Character; directory for the strategy to save weights/metrics.
#' @return Character; path to the created app directory.
#' @keywords internal
.build_flower_app <- function(recipe, app_dir = NULL, results_dir = NULL) {
  template_name <- recipe$model$template

  # Locate template in inst
  template_src <- system.file("flower_templates", template_name,
                               package = "dsFlowerClient")
  if (!nzchar(template_src) || !dir.exists(template_src)) {
    stop("Flower template not found: ", template_name,
         ". Available templates are in inst/flower_templates/.",
         call. = FALSE)
  }

  # Create app dir
  if (is.null(app_dir)) {
    app_dir <- file.path(tempdir(), "dsflower_app", template_name)
  }
  if (dir.exists(app_dir)) unlink(app_dir, recursive = TRUE)
  dir.create(app_dir, recursive = TRUE, showWarnings = FALSE)

  # Copy the Python package directory (e.g., sklearn_logreg/)
  pkg_src <- file.path(template_src, template_name)
  if (dir.exists(pkg_src)) {
    file.copy(pkg_src, app_dir, recursive = TRUE)
  }

  # Generate pyproject.toml
  .write_pyproject_toml(app_dir, recipe, results_dir = results_dir)

  app_dir
}

#' Generate pyproject.toml for a Flower App
#'
#' @param app_dir Character; path to the app directory.
#' @param recipe A \code{dsflower_recipe} object.
#' @param results_dir Character or NULL; directory for model/metrics output.
#' @return Invisible NULL.
#' @keywords internal
.write_pyproject_toml <- function(app_dir, recipe, results_dir = NULL) {
  template_name <- recipe$model$template

  # Build run_config section
  config_lines <- c(
    paste0('num-server-rounds = ', recipe$num_rounds),
    paste0('task-type = "', recipe$task$type, '"')
  )

  if (!is.null(results_dir)) {
    config_lines <- c(config_lines,
      paste0('results-dir = "', results_dir, '"'))
  }

  # Add model params
  for (nm in names(recipe$model$params)) {
    val <- recipe$model$params[[nm]]
    config_lines <- c(config_lines, .toml_kv(nm, val))
  }

  # Add strategy params
  config_lines <- c(config_lines,
                    paste0('strategy = "', recipe$strategy$name, '"'))
  for (nm in names(recipe$strategy$params)) {
    val <- recipe$strategy$params[[nm]]
    config_lines <- c(config_lines, .toml_kv(paste0("strategy-", nm), val))
  }

  # Add privacy params
  config_lines <- c(config_lines,
                    paste0('privacy-mode = "', recipe$privacy$mode, '"'))
  for (nm in names(recipe$privacy$params)) {
    val <- recipe$privacy$params[[nm]]
    config_lines <- c(config_lines, .toml_kv(paste0("privacy-", nm), val))
  }

  deps <- .template_dependencies(recipe$model$framework)

  toml <- paste0(
    '[build-system]\n',
    'requires = ["hatchling"]\n',
    'build-backend = "hatchling.build"\n\n',
    '[project]\n',
    'name = "', template_name, '"\n',
    'version = "0.1.0"\n',
    'dependencies = [', paste0('"', deps, '"', collapse = ", "), ']\n\n',
    '[tool.hatch.build.targets.wheel]\n',
    'packages = ["."]\n\n',
    '[tool.flwr.app]\n',
    'publisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "', template_name, '.server_app:app"\n',
    'clientapp = "', template_name, '.client_app:app"\n\n',
    '[tool.flwr.app.config]\n',
    paste(config_lines, collapse = "\n"), '\n'
  )

  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  invisible(NULL)
}

#' Format a key-value pair for TOML
#' @param key Character; the TOML key.
#' @param val The value.
#' @return Character; TOML line.
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

#' Get pip dependencies for a framework
#' @param framework Character; "sklearn" or "pytorch".
#' @return Character vector of pip package names.
#' @keywords internal
.template_dependencies <- function(framework) {
  base <- c("flwr[app]>=1.13.0", "numpy>=1.21.0", "pandas>=1.3.0")

  if (framework == "sklearn") {
    return(c(base, "scikit-learn>=1.0.0"))
  }
  if (framework == "pytorch") {
    return(c(base, "torch>=2.0.0"))
  }

  base
}
