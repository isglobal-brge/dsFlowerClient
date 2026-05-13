script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

run_checked <- function(command, args, error_message) {
  status <- system2(command, args, stdout = TRUE, stderr = TRUE)
  code <- attr(status, "status") %||% 0L
  if (!identical(as.integer(code), 0L)) {
    stop(error_message, "\n", paste(status, collapse = "\n"), call. = FALSE)
  }
  invisible(status)
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

make_pattern_png <- function(path, label, site, index, size = 64L) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  xs <- rep(seq(0, 1, length.out = size), each = size)
  ys <- rep(seq(0, 1, length.out = size), times = size)
  noise <- matrix(stats::runif(size * size, min = 0, max = 0.10), size, size)

  if (identical(as.integer(label), 0L)) {
    red <- matrix(xs, size, size) * 0.35 + noise
    green <- matrix(ys, size, size) * 0.25 + 0.10
    blue <- 0.70 - red * 0.30
  } else {
    cx <- 0.45 + site * 0.03
    cy <- 0.50 + (index %% 5L) * 0.015
    radius <- sqrt((xs - cx)^2 + (ys - cy)^2)
    blob <- matrix(as.numeric(radius < 0.27), size, size)
    red <- 0.20 + blob * 0.65 + noise
    green <- 0.15 + blob * 0.25
    blue <- 0.20 + matrix(ys, size, size) * 0.20
  }

  rgb_arr <- array(0, dim = c(size, size, 3L))
  rgb_arr[, , 1L] <- pmin(pmax(red, 0), 1)
  rgb_arr[, , 2L] <- pmin(pmax(green, 0), 1)
  rgb_arr[, , 3L] <- pmin(pmax(blue, 0), 1)

  grDevices::png(path, width = size, height = size)
  old_par <- graphics::par(mar = c(0, 0, 0, 0))
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  }, add = TRUE)
  graphics::plot.new()
  graphics::rasterImage(grDevices::as.raster(rgb_arr), 0, 0, 1, 1)
  invisible(path)
}

make_site_images <- function(local_root, site, n_per_site, seed) {
  set.seed(seed + site)
  site_dir <- file.path(local_root, paste0("site", site))
  dir.create(site_dir, recursive = TRUE, showWarnings = FALSE)

  labels <- rep(c(0L, 1L), length.out = n_per_site)
  labels <- sample(labels, length(labels))
  rows <- vector("list", n_per_site)

  for (i in seq_len(n_per_site)) {
    filename <- sprintf("img_%03d.png", i)
    rel_path <- file.path(paste0("site", site), filename)
    make_pattern_png(
      file.path(local_root, rel_path),
      label = labels[[i]],
      site = site,
      index = i
    )
    rows[[i]] <- data.frame(
      id = sprintf("site%d_img_%03d", site, i),
      relative_path = rel_path,
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
  run_checked("docker", c("cp", site_src, paste0(container, ":", server_root, "/")),
              paste0("Could not copy demo images into container ", container))
  invisible(TRUE)
}

cfg <- demo_config("direct_image_resnet_smoke", default_rounds = 1L)
cfg$profile <- demo_env("DSFLOWER_IMAGE_DEMO_PRIVACY", "sandbox_open")

n_sites <- length(cfg$urls)
n_per_site <- as.integer(demo_env("DSFLOWER_IMAGE_DEMO_N_PER_SITE", 8L))
server_root <- demo_env("DSFLOWER_IMAGE_DEMO_SERVER_ROOT",
                        "/tmp/dsflower_direct_image_smoke")
local_root <- demo_env(
  "DSFLOWER_IMAGE_DEMO_LOCAL_ROOT",
  file.path(tempdir(), "dsflower_direct_image_smoke")
)
containers <- trimws(strsplit(
  demo_env("DSFLOWER_IMAGE_DEMO_DOCKER_CONTAINERS",
           paste0("opal", seq_len(n_sites), "-rock", collapse = ",")),
  ",", fixed = TRUE
)[[1]])
skip_docker_copy <- identical(tolower(demo_env("DSFLOWER_IMAGE_DEMO_SKIP_DOCKER_COPY", "false")), "true")

if (n_per_site < 4L) {
  stop("DSFLOWER_IMAGE_DEMO_N_PER_SITE must be >= 4 for both classes.", call. = FALSE)
}
if (!skip_docker_copy && length(containers) != n_sites) {
  stop("DSFLOWER_IMAGE_DEMO_DOCKER_CONTAINERS must have one container per Opal URL.",
       call. = FALSE)
}

message("Generating direct-image demo PNGs under ", local_root)
site_tables <- vector("list", n_sites)
for (i in seq_len(n_sites)) {
  site_tables[[i]] <- make_site_images(local_root, i, n_per_site, cfg$seed)
}

if (!skip_docker_copy) {
  if (!nzchar(Sys.which("docker"))) {
    stop("Docker is required for the default local image demo copy. ",
         "Set DSFLOWER_IMAGE_DEMO_SKIP_DOCKER_COPY=true if images are already ",
         "available on each Rock under DSFLOWER_IMAGE_DEMO_SERVER_ROOT.",
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
  set_privacy_profile(opal, cfg$profile)
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

message("Connecting to ", n_sites, " Opal nodes for direct-image ResNet smoke demo...")
conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

recipe <- ds.flower.recipe(
  model = ds.flower.model.pytorch_resnet18(
    n_classes = 2L,
    learning_rate = as.numeric(demo_env("DSFLOWER_IMAGE_DEMO_LR", "0.001")),
    batch_size = as.integer(demo_env("DSFLOWER_IMAGE_DEMO_BATCH_SIZE", "4")),
    local_epochs = as.integer(demo_env("DSFLOWER_IMAGE_DEMO_LOCAL_EPOCHS", "1"))
  ),
  strategy = ds.flower.strategy.fedavg(),
  privacy = ds.flower.privacy(cfg$profile),
  target = "label",
  num_rounds = cfg$rounds
)

run_config <- c(
  recipe$model$params,
  recipe$strategy$params,
  list(
    data_type = "image",
    num_rounds = recipe$num_rounds,
    target_column = "label"
  )
)

message("Preparing image manifests on the servers...")
with_ds_errors(
  ds.flower.nodes.init(conns, data = "D", symbol = "flower")
)
with_ds_errors(
  ds.flower.nodes.prepare(
    conns,
    symbol = "flower",
    target_column = "label",
    feature_columns = NULL,
    run_config = run_config,
    privacy = recipe$privacy,
    template_name = recipe$model$template
  )
)

started_superlink <- FALSE
if (!isTRUE(ds.flower.superlink.status()$running)) {
  ds.flower.superlink.start(detached = FALSE)
  started_superlink <- TRUE
}
on.exit({
  try(ds.flower.nodes.cleanup(conns, symbol = "flower"), silent = TRUE)
  if (started_superlink) try(ds.flower.superlink.stop(), silent = TRUE)
}, add = TRUE)

message("Starting SuperNodes and running one federated image round...")
with_ds_errors(
  ds.flower.nodes.ensure(conns, symbol = "flower",
                         template_name = recipe$model$template)
)
run <- with_ds_errors(
  ds.flower.run.start(recipe, conns = conns, verbose = TRUE)
)

post_caps <- tryCatch(
  DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
  error = function(e) NULL
)
validate_run(run, post_caps)

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(cfg$output_root,
                     paste0("direct_image_resnet_smoke_", timestamp))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary <- list(
  demo_id = "direct_image_resnet_smoke",
  dataset_label = "Synthetic direct PNG image ResNet smoke demo",
  data_mode = "server-side PNG files + Opal metadata table",
  n_sites = n_sites,
  n_per_site = n_per_site,
  n_total = n_sites * n_per_site,
  server_root = server_root,
  local_root = normalizePath(local_root, mustWork = FALSE),
  containers = if (skip_docker_copy) NULL else containers,
  rounds = cfg$rounds,
  model = recipe$model$name,
  privacy = recipe$privacy$mode,
  status = run$status,
  model_id = run$model_id,
  flower_output_dir = run$output_dir %||% NA_character_,
  history = safe_history(run)
)

jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                     auto_unbox = TRUE, pretty = TRUE, na = "null")
saveRDS(list(summary = summary, run = run), file.path(out_dir, "workflow.rds"))

cat("\nDirect image ResNet smoke demo\n")
cat("Images: ", summary$n_total, " PNG files across ", n_sites, " sites\n", sep = "")
cat("Model: ", recipe$model$name, " | Privacy: ", recipe$privacy$mode, "\n", sep = "")
print(safe_history(run))
cat("Output: ", normalizePath(out_dir), "\n", sep = "")
cat("PASS\n")

invisible(summary)
