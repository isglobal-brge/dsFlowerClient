#!/usr/bin/env Rscript

file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE)
} else {
  tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
}
if (is.null(script_file) || !nzchar(script_file)) {
  script_file <- "inst/demos/validate_vision_methods.R"
}
demo_lib <- file.path(dirname(script_file), "lib", "benchmark_utils.R")
if (!file.exists(demo_lib)) {
  demo_lib <- file.path("inst", "demos", "lib", "benchmark_utils.R")
}
source(demo_lib)

run_checked <- function(command, args, error_message) {
  status <- system2(command, args, stdout = TRUE, stderr = TRUE)
  code <- attr(status, "status") %||% 0L
  if (!identical(as.integer(code), 0L)) {
    stop(error_message, "\n", paste(status, collapse = "\n"), call. = FALSE)
  }
  invisible(status)
}

find_validation_python <- function() {
  candidates <- c(
    demo_env("DSFLOWER_VALIDATION_PYTHON"),
    "/opt/homebrew/opt/python@3.11/libexec/bin/python",
    Sys.which("python3"),
    Sys.which("python")
  )
  candidates <- unique(candidates[nzchar(candidates)])
  for (candidate in candidates) {
    if (file.exists(candidate) || nzchar(Sys.which(candidate))) return(candidate)
  }
  stop("No Python interpreter found. Set DSFLOWER_VALIDATION_PYTHON.", call. = FALSE)
}

vision_baseline_script <- function() {
  pkg_path <- system.file("python", "vision_validation_baselines.py",
                          package = "dsFlowerClient")
  if (nzchar(pkg_path)) return(pkg_path)
  local_path <- file.path("inst", "python", "vision_validation_baselines.py")
  if (file.exists(local_path)) return(local_path)
  stop("Cannot find vision_validation_baselines.py.", call. = FALSE)
}

with_ds_errors <- function(expr) {
  tryCatch(
    force(expr),
    error = function(e) {
      errs <- tryCatch(DSI::datashield.errors(), error = function(e2) NULL)
      if (!is.null(errs)) {
        message("DataSHIELD errors:")
        utils::capture.output(print(errs)) |>
          paste(collapse = "\n") |>
          message()
      }
      stop(e)
    }
  )
}

history_metric <- function(hist, name) {
  if (is.null(hist) || !NROW(hist)) return(NA_real_)
  if (name %in% names(hist)) {
    vals <- suppressWarnings(as.numeric(hist[[name]]))
    vals <- vals[is.finite(vals)]
    if (length(vals)) return(tail(vals, 1L))
  }
  NA_real_
}

as_scalar <- function(x, default = NA_real_) {
  if (is.null(x) || length(x) == 0L) return(default)
  x <- suppressWarnings(as.numeric(x))
  if (length(x) == 0L || is.na(x[[1L]])) default else x[[1L]]
}

acceptance_params <- function(spec) {
  acc <- spec$acceptance %||% list()
  list(
    max_loss_ratio = as_scalar(acc$max_loss_ratio, 2.5),
    max_loss_margin = as_scalar(acc$max_loss_margin, 0.25)
  )
}

acceptable_federated_loss <- function(spec, central_loss, fed_loss,
                                      n_failures) {
  params <- acceptance_params(spec)
  if (!is.finite(central_loss) || !is.finite(fed_loss)) return(FALSE)
  failures <- suppressWarnings(as.numeric(n_failures))
  no_failures <- length(failures) == 0L || is.na(failures[[1L]]) ||
    failures[[1L]] <= 0
  no_failures &&
    fed_loss <= central_loss * params$max_loss_ratio + params$max_loss_margin
}

make_png_pair <- function(image_path, mask_path, label, site, index,
                          size = 64L) {
  dir.create(dirname(image_path), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(mask_path), recursive = TRUE, showWarnings = FALSE)

  grid <- seq(0, 1, length.out = size)
  xs <- rep(grid, each = size)
  ys <- rep(grid, times = size)
  noise <- matrix(stats::runif(size * size, min = 0, max = 0.08), size, size)

  cx <- 0.35 + 0.08 * (site %% 3)
  cy <- 0.42 + 0.02 * (index %% 7)
  radius <- sqrt((xs - cx)^2 + (ys - cy)^2)
  blob <- matrix(as.numeric(radius < 0.22), size, size)

  if (identical(as.integer(label), 1L)) {
    red <- 0.20 + blob * 0.70 + noise
    green <- 0.15 + blob * 0.35
    blue <- matrix(ys, size, size) * 0.35
    mask <- blob
  } else {
    red <- matrix(xs, size, size) * 0.35 + noise
    green <- matrix(ys, size, size) * 0.25 + 0.12
    blue <- 0.65 - red * 0.25
    mask <- matrix(0, size, size)
  }

  rgb_arr <- array(0, dim = c(size, size, 3L))
  rgb_arr[, , 1L] <- pmin(pmax(red, 0), 1)
  rgb_arr[, , 2L] <- pmin(pmax(green, 0), 1)
  rgb_arr[, , 3L] <- pmin(pmax(blue, 0), 1)

  grDevices::png(image_path, width = size, height = size)
  old_par <- graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::rasterImage(grDevices::as.raster(rgb_arr), 0, 0, 1, 1)
  graphics::par(old_par)
  grDevices::dev.off()

  grDevices::png(mask_path, width = size, height = size)
  old_par2 <- graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::rasterImage(grDevices::as.raster(mask), 0, 0, 1, 1)
  graphics::par(old_par2)
  grDevices::dev.off()

  invisible(TRUE)
}

make_site_images <- function(local_root, site, n_per_site, seed, size) {
  set.seed(seed + site)
  labels <- rep(c(0L, 1L), length.out = n_per_site)
  labels <- sample(labels, length(labels))
  rows <- vector("list", n_per_site)

  for (i in seq_len(n_per_site)) {
    image_rel <- file.path(paste0("site", site), "images",
                           sprintf("img_%03d.png", i))
    mask_rel <- file.path(paste0("site", site), "masks",
                          sprintf("mask_%03d.png", i))
    make_png_pair(
      file.path(local_root, image_rel),
      file.path(local_root, mask_rel),
      label = labels[[i]],
      site = site,
      index = i,
      size = size
    )
    rows[[i]] <- data.frame(
      id = sprintf("vision_site%d_%03d", site, i),
      relative_path = image_rel,
      mask_path = mask_rel,
      label = as.integer(labels[[i]]),
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, rows)
}

copy_site_images_to_container <- function(local_root, site, container, server_root) {
  if (!nzchar(container)) return(invisible(FALSE))
  site_name <- paste0("site", site)
  site_src <- file.path(local_root, site_name)
  run_checked("docker", c("exec", container, "mkdir", "-p", server_root),
              paste0("Could not create image root in container ", container))
  run_checked("docker", c("exec", container, "rm", "-rf",
                          file.path(server_root, site_name)),
              paste0("Could not clean previous images in container ", container))
  run_checked("docker", c("cp", site_src, paste0(container, ":", server_root, "/")),
              paste0("Could not copy demo images into container ", container))
  invisible(TRUE)
}

vision_specs <- function() {
  common <- list(n_classes = 2L, learning_rate = 0.001,
                 batch_size = 2L, local_epochs = 1L,
                 image_size = 64L)
  list(
    list(method = "pytorch_resnet18",
         model = do.call(ds.flower.model.pytorch_resnet18, common),
         task = "classification", target = "label",
         notes = "Synthetic PNG image classification with ResNet-18."),
    list(method = "pytorch_densenet121",
         model = do.call(ds.flower.model.pytorch_densenet121, common),
         task = "classification", target = "label",
         notes = "Synthetic PNG image classification with DenseNet-121.")
  )
}

selected_specs <- function(specs) {
  selection <- demo_env("DSFLOWER_VALIDATE_VISION_METHODS", "all")
  if (!nzchar(selection) || identical(tolower(selection), "all")) return(specs)
  wanted <- trimws(strsplit(selection, ",", fixed = TRUE)[[1]])
  specs[vapply(specs, function(x) x$method %in% wanted, logical(1))]
}

run_central_baseline <- function(spec, samples_csv, image_root, out_dir,
                                 python, baseline_script, rounds, seed) {
  out_json <- file.path(out_dir, paste0(spec$method, "_centralized.json"))
  params_json <- file.path(out_dir, paste0(spec$method, "_params.json"))
  jsonlite::write_json(spec$model$params, params_json,
                       auto_unbox = TRUE, pretty = TRUE, null = "null")
  args <- c(
    baseline_script,
    "--method", spec$method,
    "--samples", samples_csv,
    "--image-root", image_root,
    "--target-col", spec$target,
    "--path-col", "relative_path",
    "--mask-path-col", "mask_path",
    "--model-params-file", params_json,
    "--rounds", as.character(rounds),
    "--seed", as.character(seed),
    "--output", out_json
  )
  res <- system2(python, args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(res, "status") %||% 0L
  if (!identical(as.integer(status), 0L)) {
    return(list(status = "failed", error = paste(res, collapse = "\n")))
  }
  parsed <- jsonlite::fromJSON(out_json, simplifyVector = FALSE)
  parsed$status <- parsed$status %||% "ok"
  parsed
}

run_federated_vision <- function(spec, conns, rounds) {
  run <- with_ds_errors(
    ds.flower.fit(
      conns, symbol = "D", target = spec$target, model = spec$model,
      rounds = rounds, data_kind = "image", verbose = TRUE)
  )
  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
    error = function(e) NULL
  )
  validate_run(run, post_caps)
  list(
    status = if (identical(as.integer(run$status), 0L)) "ok" else "failed",
    loss = history_metric(safe_history(run), "loss"),
    n_failures = history_metric(safe_history(run), "n_failures"),
    output_dir = run$output_dir,
    history = safe_history(run)
  )
}

cfg <- demo_config("vision_method_validation", default_rounds = 1L)

if (!nzchar(Sys.getenv("DSFLOWER_ARTIFACT_WATCHDOG_GRACE_SECS"))) {
  Sys.setenv(DSFLOWER_ARTIFACT_WATCHDOG_GRACE_SECS = "1")
}

n_sites <- length(cfg$urls)
n_per_site <- as.integer(demo_env("DSFLOWER_VISION_VALIDATION_N_PER_SITE", 4L))
image_size <- as.integer(demo_env("DSFLOWER_VISION_VALIDATION_IMAGE_SIZE", 64L))
server_root <- demo_env("DSFLOWER_VISION_VALIDATION_SERVER_ROOT",
                        "/tmp/dsflower_vision_method_validation")
local_root <- demo_env(
  "DSFLOWER_VISION_VALIDATION_LOCAL_ROOT",
  file.path(tempdir(), "dsflower_vision_method_validation")
)
containers <- trimws(strsplit(
  demo_env("DSFLOWER_VISION_VALIDATION_DOCKER_CONTAINERS",
           paste0("opal", seq_len(n_sites), "-rock", collapse = ",")),
  ",", fixed = TRUE
)[[1]])
skip_docker_copy <- identical(
  tolower(demo_env("DSFLOWER_VISION_VALIDATION_SKIP_DOCKER_COPY", "false")),
  "true"
)

if (n_per_site < 4L) {
  stop("DSFLOWER_VISION_VALIDATION_N_PER_SITE must be >= 4.", call. = FALSE)
}
if (!skip_docker_copy && length(containers) != n_sites) {
  stop("DSFLOWER_VISION_VALIDATION_DOCKER_CONTAINERS must have one container per Opal URL.",
       call. = FALSE)
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(
  demo_env("DSFLOWER_VISION_VALIDATION_OUTPUT_ROOT",
           file.path(getwd(), "dsflower_output", "vision_method_validation")),
  timestamp
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Generating synthetic vision validation PNGs under ", local_root)
site_tables <- vector("list", n_sites)
for (i in seq_len(n_sites)) {
  site_tables[[i]] <- make_site_images(local_root, i, n_per_site,
                                       cfg$seed, image_size)
}
pooled <- do.call(rbind, site_tables)
samples_csv <- file.path(out_dir, "pooled_samples.csv")
utils::write.csv(pooled, samples_csv, row.names = FALSE)

if (!skip_docker_copy) {
  if (!nzchar(Sys.which("docker"))) {
    stop("Docker is required for the default local vision validation copy. ",
         "Set DSFLOWER_VISION_VALIDATION_SKIP_DOCKER_COPY=true if files are ",
         "already visible to each Rock under DSFLOWER_VISION_VALIDATION_SERVER_ROOT.",
         call. = FALSE)
  }
  for (i in seq_len(n_sites)) {
    message("Copying site ", i, " images to ", containers[[i]], ":", server_root)
    copy_site_images_to_container(local_root, i, containers[[i]], server_root)
  }
}

table_paths <- character(n_sites)
for (i in seq_len(n_sites)) {
  opal <- opal_login(cfg, i)
  on.exit(try(opalr::opal.logout(opal), silent = TRUE), add = TRUE)
  ensure_project(opal, cfg$project)
  opalr::dsadmin.set_option(opal, "dsflower.image_data_root",
                            server_root, profile = "default")
  table_name <- paste0(cfg$table_prefix, "_site", i)
  table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name,
                                        site_tables[[i]])
  opalr::opal.logout(opal)
}

builder <- DSI::newDSLoginBuilder()
for (i in seq_len(n_sites)) {
  builder$append(
    server = paste0("opal", i),
    url = cfg$urls[[i]],
    user = cfg$users[[i]],
    password = cfg$passwords[[i]],
    table = table_paths[[i]],
    driver = "OpalDriver"
  )
}

message("Connecting to ", n_sites, " Opal nodes for vision method validation...")
conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

started_superlink <- FALSE
if (!isTRUE(ds.flower.superlink.status()$running)) {
  ds.flower.superlink.start(detached = FALSE)
  started_superlink <- TRUE
}
on.exit({
  if (started_superlink) try(ds.flower.superlink.stop(), silent = TRUE)
}, add = TRUE)

python <- find_validation_python()
baseline_script <- vision_baseline_script()
specs <- selected_specs(vision_specs())

results <- vector("list", length(specs))
for (i in seq_along(specs)) {
  spec <- specs[[i]]
  message("\n=== Validating ", spec$method, " ===")
  central <- run_central_baseline(
    spec, samples_csv, local_root, out_dir, python, baseline_script,
    rounds = cfg$rounds, seed = cfg$seed
  )
  fed <- tryCatch(
    run_federated_vision(spec, conns, rounds = cfg$rounds),
    error = function(e) list(status = "failed", error = conditionMessage(e),
                             loss = NA_real_, n_failures = NA_real_)
  )

  central_loss <- suppressWarnings(as.numeric(central$loss %||% NA_real_))
  fed_loss <- suppressWarnings(as.numeric(fed$loss %||% NA_real_))
  fed_n_failures <- suppressWarnings(as.numeric(fed$n_failures %||% NA_real_))
  acceptance <- acceptance_params(spec)
  acceptable_loss <- acceptable_federated_loss(
    spec, central_loss, fed_loss, fed_n_failures
  )
  results[[i]] <- data.frame(
    method = spec$method,
    task = spec$task,
    target = spec$target,
    rounds = cfg$rounds,
    n_sites = n_sites,
    n_per_site = n_per_site,
    n_total = n_sites * n_per_site,
    centralized_status = central$status %||% "unknown",
    centralized_metric = central$metric_name %||% NA_character_,
    centralized_loss = central_loss,
    centralized_accuracy = suppressWarnings(as.numeric(central$accuracy %||% NA_real_)),
    centralized_dice = suppressWarnings(as.numeric(central$dice %||% NA_real_)),
    federated_status = fed$status %||% "unknown",
    federated_loss = fed_loss,
    federated_n_failures = fed_n_failures,
    delta_loss = fed_loss - central_loss,
    acceptance_max_loss_ratio = acceptance$max_loss_ratio,
    acceptance_max_loss_margin = acceptance$max_loss_margin,
    acceptable_loss = acceptable_loss,
    validation_status = if (identical(central$status, "ok") &&
                            identical(fed$status, "ok") &&
                            isTRUE(acceptable_loss)) "pass" else "failed",
    error = fed$error %||% central$error %||% "",
    notes = spec$notes,
    stringsAsFactors = FALSE
  )
}

results_df <- do.call(rbind, results)
payload <- list(
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
  demo_id = "vision_method_validation",
  n_sites = n_sites,
  n_per_site = n_per_site,
  n_total = n_sites * n_per_site,
  image_size = image_size,
  server_root = server_root,
  local_root = normalizePath(local_root, mustWork = FALSE),
  results = results_df
)
json_path <- file.path(out_dir, "vision_method_validation_results.json")
jsonlite::write_json(payload, json_path, auto_unbox = TRUE, pretty = TRUE,
                     na = "null")
utils::write.csv(results_df, file.path(out_dir, "vision_method_validation_results.csv"),
                 row.names = FALSE)

extdata_dir <- file.path("inst", "extdata")
if (dir.exists(extdata_dir)) {
  file.copy(json_path,
            file.path(extdata_dir, "dsflower_vision_validation_results.json"),
            overwrite = TRUE)
}

print(results_df)
cat("Output: ", normalizePath(out_dir), "\n", sep = "")
if (any(results_df$validation_status != "pass")) {
  stop("One or more vision validation methods failed.", call. = FALSE)
}
cat("PASS\n")

invisible(payload)
