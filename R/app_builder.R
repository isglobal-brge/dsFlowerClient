# Module: Flower App Generation
# Builds Flower App from recipe specs using templates fetched from the server.

#' Build a Flower App from a recipe
#'
#' Fetches the template from the server (via DataSHIELD), writes the Python
#' files locally, and generates a \code{pyproject.toml} with run_config
#' from the recipe.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param conns DSI connections object (used to fetch templates from server).
#' @param app_dir Character; directory to create the app in (default: tempdir).
#' @param results_dir Character; directory for the strategy to save weights/metrics.
#' @return Character; path to the created app directory.
#' @keywords internal
.build_flower_app <- function(recipe, conns, app_dir = NULL,
                               results_dir = NULL) {
  template_name <- recipe$model$template

  # Fetch template from the server
  template_files <- .fetch_template(conns, template_name)

  # Create app dir
  if (is.null(app_dir)) {
    app_dir <- file.path(tempdir(), "dsflower_app", template_name)
  }
  if (dir.exists(app_dir)) unlink(app_dir, recursive = TRUE)
  dir.create(app_dir, recursive = TRUE, showWarnings = FALSE)

  # Write template files to disk
  for (rel_path in names(template_files)) {
    full_path <- file.path(app_dir, rel_path)
    dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
    writeLines(template_files[[rel_path]], full_path)
  }

  # Generate pyproject.toml
  .write_pyproject_toml(app_dir, recipe, results_dir = results_dir)

  app_dir
}

#' Fetch a template from the server
#'
#' Calls \code{flowerGetTemplateDS} on the first available server.
#' Results are cached per session so repeated runs with the same model
#' don't re-fetch.
#'
#' @param conns DSI connections object.
#' @param template_name Character; template name.
#' @return Named list mapping relative file paths to contents.
#' @keywords internal
.fetch_template <- function(conns, template_name) {
  # Check session cache
  cache <- .dsflower_client_env$.template_cache
  if (!is.null(cache[[template_name]])) {
    return(cache[[template_name]])
  }

  # Verify all servers have this template
  all_templates <- tryCatch(
    DSI::datashield.aggregate(conns, expr = quote(flowerListTemplatesDS())),
    error = function(e) NULL
  )
  if (!is.null(all_templates)) {
    missing <- character(0)
    for (srv in names(all_templates)) {
      if (!template_name %in% all_templates[[srv]]) {
        missing <- c(missing, srv)
      }
    }
    if (length(missing) > 0) {
      stop("Template '", template_name, "' is not available on: ",
           paste(missing, collapse = ", "), ".", call. = FALSE)
    }
  }

  # Fetch template files from the first server
  srv <- names(conns)[1]
  result <- tryCatch(
    DSI::datashield.aggregate(
      conns[srv],
      expr = call("flowerGetTemplateDS", template_name)
    ),
    error = function(e) {
      stop("Failed to fetch template '", template_name, "' from server '",
           srv, "': ", conditionMessage(e), call. = FALSE)
    }
  )

  files <- result[[srv]]$files
  if (is.null(files) || length(files) == 0) {
    stop("Template '", template_name, "' returned no files from server '",
         srv, "'.", call. = FALSE)
  }

  # Cache for this session
  if (is.null(.dsflower_client_env$.template_cache)) {
    .dsflower_client_env$.template_cache <- list()
  }
  .dsflower_client_env$.template_cache[[template_name]] <- files

  files
}

#' List templates available on the servers
#'
#' Queries each server for its installed templates.
#'
#' @param conns DSI connections object.
#' @return A named list mapping server names to character vectors of template names.
#' @export
ds.flower.templates <- function(conns) {
  results <- DSI::datashield.aggregate(
    conns, expr = quote(flowerListTemplatesDS())
  )
  results
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

  # SecAgg and metric suppression flags (passed via run_config to server_app)
  # These are informational in pyproject.toml; the server enforces via manifest.
  if (recipe$privacy$mode %in% c("secure", "dp")) {
    config_lines <- c(config_lines,
      'require-secure-aggregation = true',
      'allow-per-node-metrics = false'
    )
  } else {
    config_lines <- c(config_lines,
      'require-secure-aggregation = false',
      'allow-per-node-metrics = true'
    )
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
  if (framework == "pytorch_vision") {
    return(c(base, "torch>=2.0.0", "torchvision>=0.15.0", "Pillow>=9.0.0"))
  }
  if (framework == "xgboost") {
    return(c(base, "xgboost>=1.7.0"))
  }

  base
}
