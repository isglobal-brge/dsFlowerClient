# Direct-image ResNet smoke demo (table-backed image binding).
#
# Generates a tiny synthetic 2-class PNG dataset, stages the pixels on each Rock
# server (local `docker cp`, remote `ssh`+docker, or pre-staged), registers one
# Opal samples table per site + the `dsflower.image_data_root` option, then runs
# ONE federated DP-SGD round on a frozen-backbone ResNet18 head via the unified
# ds.flower.fit() API. No credentials are hard-coded -- everything comes from
# DSFLOWER_OPAL_* / DSFLOWER_IMAGE_DEMO_* env vars (see lib/benchmark_utils.R).
#
# Example (the live workshop federation, remote staging over SSH):
#   DSFLOWER_OPAL_URLS=https://nairobi.datashield.live,https://dakar.datashield.live,https://douala.datashield.live \
#   DSFLOWER_OPAL_USERS=administrator DSFLOWER_OPAL_PASSWORDS=password \
#   DSFLOWER_IMAGE_DEMO_DOCKER_CONTAINERS=nairobi-rock,dakar-rock,douala-rock \
#   DSFLOWER_IMAGE_DEMO_SERVER_ROOT=/srv/imaging/direct_image_smoke \
#   DSFLOWER_IMAGE_DEMO_SSH_HOST=root@94.130.67.179 \
#   DSFLOWER_IMAGE_DEMO_SSH_KEY=$HOME/keys/id_ed25519 \
#   Rscript direct_image_resnet_smoke.R

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))
suppressMessages(library(dsFlowerClient))

run_checked <- function(command, args, error_message) {
  status <- system2(command, args, stdout = TRUE, stderr = TRUE)
  code <- attr(status, "status") %||% 0L
  if (!identical(as.integer(code), 0L)) {
    stop(error_message, "\n", paste(status, collapse = "\n"), call. = FALSE)
  }
  invisible(status)
}

with_ds_errors <- function(expr) {
  tryCatch(force(expr), error = function(e) {
    errs <- tryCatch(DSI::datashield.errors(), error = function(e2) NULL)
    if (!is.null(errs)) { message("DataSHIELD errors:")
      message(paste(utils::capture.output(print(errs)), collapse = "\n")) }
    stop(e)
  })
}

# ---- synthetic 2-class PNGs (R-native; no Python/PIL needed) ----
make_pattern_png <- function(path, label, site, index, size = 64L) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  xs <- rep(seq(0, 1, length.out = size), each = size)
  ys <- rep(seq(0, 1, length.out = size), times = size)
  noise <- matrix(stats::runif(size * size, 0, 0.10), size, size)
  if (identical(as.integer(label), 0L)) {
    red <- matrix(xs, size, size) * 0.35 + noise
    green <- matrix(ys, size, size) * 0.25 + 0.10
    blue <- 0.70 - red * 0.30
  } else {
    cx <- 0.45 + site * 0.03; cy <- 0.50 + (index %% 5L) * 0.015
    blob <- matrix(as.numeric(sqrt((xs - cx)^2 + (ys - cy)^2) < 0.27), size, size)
    red <- 0.20 + blob * 0.65 + noise
    green <- 0.15 + blob * 0.25
    blue <- 0.20 + matrix(ys, size, size) * 0.20
  }
  arr <- array(0, dim = c(size, size, 3L))
  arr[, , 1L] <- pmin(pmax(red, 0), 1); arr[, , 2L] <- pmin(pmax(green, 0), 1)
  arr[, , 3L] <- pmin(pmax(blue, 0), 1)
  grDevices::png(path, width = size, height = size)
  old <- graphics::par(mar = c(0, 0, 0, 0))
  on.exit({ graphics::par(old); grDevices::dev.off() }, add = TRUE)
  graphics::plot.new(); graphics::rasterImage(grDevices::as.raster(arr), 0, 0, 1, 1)
  invisible(path)
}

make_site_images <- function(local_root, site, n_per_site, seed) {
  set.seed(seed + site)
  labels <- sample(rep(c(0L, 1L), length.out = n_per_site))
  rows <- vector("list", n_per_site)
  for (i in seq_len(n_per_site)) {
    rel <- file.path(paste0("site", site), sprintf("img_%03d.png", i))
    make_pattern_png(file.path(local_root, rel), labels[[i]], site, i)
    rows[[i]] <- data.frame(id = sprintf("site%d_img_%03d", site, i),
                            relative_path = rel, label = as.integer(labels[[i]]),
                            stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

# ---- staging: local docker cp | remote ssh+docker | pre-staged (skip) ----
stage_site <- function(mode, local_root, site, container, server_root, ssh_host, ssh_key) {
  site_name <- paste0("site", site)
  if (identical(mode, "skip")) return(invisible(FALSE))
  if (identical(mode, "docker")) {
    run_checked("docker", c("exec", container, "mkdir", "-p", server_root),
                paste0("Could not create image root in container ", container))
    run_checked("docker", c("cp", file.path(local_root, site_name),
                            paste0(container, ":", server_root, "/")),
                paste0("Could not docker cp images into ", container))
    return(invisible(TRUE))
  }
  # mode == "ssh": stream a clean tar of the site dir into the remote container
  ssh_args <- paste(vapply(c("-i", ssh_key, "-o", "StrictHostKeyChecking=no",
                             "-o", "ConnectTimeout=30", ssh_host), shQuote, character(1)),
                    collapse = " ")
  run_checked("ssh", c("-i", ssh_key, "-o", "StrictHostKeyChecking=no",
                       "-o", "ConnectTimeout=30", ssh_host,
                       sprintf("docker exec %s mkdir -p %s/%s", container, server_root, site_name)),
              paste0("Could not create remote image dir in ", container))
  cmd <- paste0("COPYFILE_DISABLE=1 tar --no-mac-metadata -C ",
                shQuote(file.path(local_root, site_name)), " -cf - . | ssh ", ssh_args, " ",
                shQuote(sprintf("docker exec -i %s tar -C %s/%s -xf -", container, server_root, site_name)))
  if (!identical(as.integer(system(cmd)), 0L)) {
    stop("Remote SSH image staging failed for ", container, ".", call. = FALSE)
  }
  invisible(TRUE)
}

# --------------------------------------------------------------------------- #
cfg <- demo_config("direct_image_resnet_smoke", default_rounds = 1L)
cfg$profile <- demo_env("DSFLOWER_IMAGE_DEMO_PRIVACY", "sandbox_open")
n_sites <- length(cfg$urls)
n_per_site <- as.integer(demo_env("DSFLOWER_IMAGE_DEMO_N_PER_SITE", 8L))
server_root <- demo_env("DSFLOWER_IMAGE_DEMO_SERVER_ROOT", "/srv/imaging/direct_image_smoke")
local_root <- demo_env("DSFLOWER_IMAGE_DEMO_LOCAL_ROOT",
                       file.path(tempdir(), "dsflower_direct_image_smoke"))
containers <- trimws(strsplit(
  demo_env("DSFLOWER_IMAGE_DEMO_DOCKER_CONTAINERS",
           paste0("opal", seq_len(n_sites), "-rock", collapse = ",")), ",", fixed = TRUE)[[1]])
ssh_host <- demo_env("DSFLOWER_IMAGE_DEMO_SSH_HOST", "")
ssh_key <- demo_env("DSFLOWER_IMAGE_DEMO_SSH_KEY", "")
stage_mode <- if (demo_bool("DSFLOWER_IMAGE_DEMO_SKIP_DOCKER_COPY", FALSE)) {
  "skip"
} else if (nzchar(ssh_host)) {
  "ssh"
} else {
  "docker"
}

if (n_per_site < 4L) stop("DSFLOWER_IMAGE_DEMO_N_PER_SITE must be >= 4.", call. = FALSE)
if (!identical(stage_mode, "skip") && length(containers) != n_sites) {
  stop("DSFLOWER_IMAGE_DEMO_DOCKER_CONTAINERS must list one container per Opal URL.", call. = FALSE)
}

message("Generating synthetic image PNGs under ", local_root, " (stage mode: ", stage_mode, ")")
site_tables <- lapply(seq_len(n_sites), function(i) make_site_images(local_root, i, n_per_site, cfg$seed))

for (i in seq_len(n_sites)) {
  if (!identical(stage_mode, "skip"))
    message("Staging site ", i, " images -> ", containers[[i]], ":", server_root)
  stage_site(stage_mode, local_root, i, containers[[i]], server_root, ssh_host, ssh_key)
}

table_paths <- character(n_sites)
for (i in seq_len(n_sites)) {
  opal <- opal_login(cfg, i)
  ensure_project(opal, cfg$project)
  set_privacy_profile(opal, cfg$profile)
  opalr::dsadmin.set_option(opal, "dsflower.image_data_root", server_root, profile = "default")
  table_paths[[i]] <- upload_site_table(opal, cfg$project,
                                        paste0(cfg$table_prefix, "_site", i), site_tables[[i]])
  opalr::opal.logout(opal)
}

builder <- DSI::newDSLoginBuilder()
for (i in seq_len(n_sites)) builder$append(server = paste0("opal", i), url = cfg$urls[[i]],
  user = cfg$users[[i]], password = cfg$passwords[[i]], table = table_paths[[i]], driver = "OpalDriver")
message("Connecting to ", n_sites, " Opal nodes...")
conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

# Unified API: ds.flower.fit owns connect/prepare(image)/run/cleanup. The resnet
# model name marks the run as image, so a table-backed samples collection (with a
# relative_path column + dsflower.image_data_root) is staged + trained as image.
message("Running one federated image round via ds.flower.fit ...")
run <- with_ds_errors(ds.flower.fit(
  conns, symbol = "D", target = "label", model = "pytorch_resnet18",
  model_params = list(
    n_classes = 2L,
    image_size = as.integer(demo_env("DSFLOWER_IMAGE_DEMO_IMAGE_SIZE", "224")),
    learning_rate = as.numeric(demo_env("DSFLOWER_IMAGE_DEMO_LR", "0.001")),
    batch_size = as.integer(demo_env("DSFLOWER_IMAGE_DEMO_BATCH_SIZE", "4")),
    local_epochs = as.integer(demo_env("DSFLOWER_IMAGE_DEMO_LOCAL_EPOCHS", "1"))),
  rounds = cfg$rounds, verbose = TRUE))

stopifnot(!is.null(run), identical(as.integer(run$status), 0L))

out_dir <- file.path(cfg$output_root,
                     paste0("direct_image_resnet_smoke_", format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary <- list(demo_id = "direct_image_resnet_smoke",
                dataset_label = "Synthetic direct PNG image ResNet smoke (table binding)",
                stage_mode = stage_mode, n_sites = n_sites, n_per_site = n_per_site,
                n_total = n_sites * n_per_site, server_root = server_root,
                rounds = cfg$rounds, model = "pytorch_resnet18", status = run$status,
                history = safe_history(run))
jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                     auto_unbox = TRUE, pretty = TRUE, na = "null")

cat("\nDirect image ResNet smoke demo\n")
cat("Images: ", summary$n_total, " PNGs across ", n_sites, " sites | stage: ", stage_mode, "\n", sep = "")
cat("Model: pytorch_resnet18 | rounds: ", cfg$rounds, " | status: ", run$status, "\n", sep = "")
cat("Output: ", normalizePath(out_dir), "\nPASS\n", sep = "")
invisible(summary)
