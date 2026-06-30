#!/usr/bin/env Rscript

file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE)
} else {
  tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
}
if (is.null(script_file) || !nzchar(script_file)) {
  script_file <- "inst/demos/benchmark_pathmnist_direct_image.R"
}
demo_lib <- file.path(dirname(script_file), "lib", "benchmark_utils.R")
if (!file.exists(demo_lib)) {
  demo_lib <- file.path("inst", "demos", "lib", "benchmark_utils.R")
}
source(demo_lib)

run_checked <- function(command, args, error_message, timeout = 900) {
  res <- processx::run(command, args, echo = TRUE, timeout = timeout,
                       error_on_status = FALSE)
  if (!identical(as.integer(res$status), 0L)) {
    stop(error_message, "\n", res$stdout, "\n", res$stderr, call. = FALSE)
  }
  invisible(res)
}

ensure_python_module <- function(module, package_spec = module) {
  dsFlowerClient:::.ensure_client_framework("pytorch_vision")
  python <- dsFlowerClient:::.client_python_cmd()
  check <- processx::run(python, c("-c", sprintf("import %s", module)),
                         error_on_status = FALSE)
  if (check$status == 0L) return(invisible(TRUE))

  uv <- dsFlowerClient:::.ensure_client_uv()
  install <- processx::run(
    uv,
    c("pip", "install", "--python", python, "--quiet", package_spec),
    echo_cmd = FALSE,
    echo = TRUE,
    timeout = 900
  )
  if (install$status != 0L) {
    stop("Could not install Python package: ", package_spec, call. = FALSE)
  }
  invisible(TRUE)
}

find_python <- function() {
  dsFlowerClient:::.ensure_client_framework("pytorch_vision")
  dsFlowerClient:::.client_python_cmd()
}

helper_script <- function() {
  candidates <- c(
    system.file("python", "resnet_image_table_eval.py",
                package = "dsFlowerClient"),
    file.path("inst", "python", "resnet_image_table_eval.py")
  )
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (!length(candidates)) {
    stop("Cannot find inst/python/resnet_image_table_eval.py", call. = FALSE)
  }
  normalizePath(candidates[[1L]], mustWork = FALSE)
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

build_pathmnist_dataset <- function(local_root, n_sites, n_per_site, seed) {
  ensure_python_module("medmnist", "medmnist>=3.0.0")
  python <- find_python()
  out_csv <- file.path(local_root, "pooled_samples.csv")
  out_meta <- file.path(local_root, "dataset_metadata.json")
  dir.create(local_root, recursive = TRUE, showWarnings = FALSE)

  class_a <- as.integer(demo_env("DSFLOWER_PATHMNIST_CLASS_A", "0"))
  class_b <- as.integer(demo_env("DSFLOWER_PATHMNIST_CLASS_B", "8"))
  if (n_per_site %% 2L != 0L) {
    stop("DSFLOWER_PATHMNIST_N_PER_SITE must be even for balanced classes.",
         call. = FALSE)
  }

  code <- paste(
    "import json, os",
    "import numpy as np",
    "import pandas as pd",
    "from PIL import Image",
    "from medmnist import PathMNIST",
    sprintf("local_root = %s", jsonlite::toJSON(local_root, auto_unbox = TRUE)),
    sprintf("out_csv = %s", jsonlite::toJSON(out_csv, auto_unbox = TRUE)),
    sprintf("out_meta = %s", jsonlite::toJSON(out_meta, auto_unbox = TRUE)),
    sprintf("n_sites = int(%s)", jsonlite::toJSON(n_sites, auto_unbox = TRUE)),
    sprintf("n_per_site = int(%s)", jsonlite::toJSON(n_per_site, auto_unbox = TRUE)),
    sprintf("seed = int(%s)", jsonlite::toJSON(seed, auto_unbox = TRUE)),
    sprintf("classes = [int(%s), int(%s)]",
            jsonlite::toJSON(class_a, auto_unbox = TRUE),
            jsonlite::toJSON(class_b, auto_unbox = TRUE)),
    "rng = np.random.default_rng(seed)",
    "ds = PathMNIST(split='train', download=True, size=28)",
    "imgs = np.asarray(ds.imgs)",
    "labels = np.asarray(ds.labels).reshape(-1).astype(int)",
    "label_map = ds.info.get('label', {})",
    "rows = []",
    "per_class_site = n_per_site // 2",
    "for cls_idx, cls in enumerate(classes):",
    "    pool = np.where(labels == cls)[0]",
    "    if len(pool) < per_class_site * n_sites:",
    "        raise RuntimeError(f'class {cls} has only {len(pool)} examples')",
    "    picked = rng.permutation(pool)[:per_class_site * n_sites]",
    "    for site in range(1, n_sites + 1):",
    "        offset = (site - 1) * per_class_site",
    "        site_idx = picked[offset:(offset + per_class_site)]",
    "        for j, sample_idx in enumerate(site_idx):",
    "            rel = os.path.join(f'site{site}', 'images',",
    "                               f'pathmnist_c{cls}_{j + 1:04d}.png')",
    "            out_path = os.path.join(local_root, rel)",
    "            os.makedirs(os.path.dirname(out_path), exist_ok=True)",
    "            Image.fromarray(imgs[sample_idx]).save(out_path)",
    "            rows.append({",
    "                'id': f'pathmnist_s{site}_c{cls}_{j + 1:04d}',",
    "                'site': site,",
    "                'relative_path': rel,",
    "                'source_class': int(cls),",
    "                'label': int(cls_idx)",
    "            })",
    "df = pd.DataFrame(rows)",
    "df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)",
    "df.to_csv(out_csv, index=False)",
    "meta = {",
    "    'dataset': 'PathMNIST',",
    "    'source': 'MedMNIST v2 train split',",
    "    'classes': {str(c): label_map.get(str(c), label_map.get(c, str(c))) for c in classes},",
    "    'binary_mapping': {str(classes[0]): 0, str(classes[1]): 1},",
    "    'n_sites': n_sites,",
    "    'n_per_site': n_per_site,",
    "    'n_total': int(len(df)),",
    "    'seed': seed",
    "}",
    "json.dump(meta, open(out_meta, 'w'), indent=2, sort_keys=True)",
    sep = "\n"
  )
  run_checked(python, c("-c", code), "Could not build PathMNIST image dataset.",
              timeout = 1800)
  list(
    samples = utils::read.csv(out_csv, check.names = FALSE),
    metadata = jsonlite::fromJSON(out_meta, simplifyVector = FALSE),
    samples_csv = out_csv
  )
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
              paste0("Could not copy demo images into container ", container),
              timeout = 1200)
  invisible(TRUE)
}

set_image_options <- function(cfg, server_root) {
  for (i in seq_along(cfg$urls)) {
    opal <- opal_login(cfg, i)
    on.exit(try(opalr::opal.logout(opal), silent = TRUE), add = TRUE)
    opalr::dsadmin.set_option(opal, "dsflower.image_data_root",
                              server_root, profile = "default")
    opalr::opal.logout(opal)
  }
  invisible(TRUE)
}

connect_tables <- function(cfg, table_paths) {
  builder <- DSI::newDSLoginBuilder()
  for (i in seq_along(cfg$urls)) {
    builder$append(
      server = paste0("opal", i),
      url = cfg$urls[[i]],
      user = cfg$users[[i]],
      password = cfg$passwords[[i]],
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }
  DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
}

run_central_resnet <- function(samples_csv, image_root, model_params,
                               rounds, seed, out_dir) {
  python <- find_python()
  helper <- helper_script()
  out_json <- file.path(out_dir, "centralized_resnet18.json")
  args <- c(
    helper,
    "--samples", samples_csv,
    "--image-root", image_root,
    "--target-col", "label",
    "--path-col", "relative_path",
    "--output", out_json,
    "--epochs", as.character(rounds * as.integer(model_params$local_epochs %||% 1L)),
    "--batch-size", as.character(model_params$batch_size),
    "--lr", as.character(model_params$learning_rate),
    "--image-size", as.character(model_params$image_size),
    "--n-classes", as.character(model_params$n_classes),
    "--seed", as.character(seed),
    "--threads", as.character(model_params$threads %||% 2L)
  )
  run_checked(python, args, "Centralized PathMNIST ResNet baseline failed.",
              timeout = 1800)
  jsonlite::fromJSON(out_json, simplifyVector = FALSE)
}

evaluate_checkpoint <- function(samples_csv, image_root, checkpoint,
                                model_params, seed, out_dir) {
  python <- find_python()
  helper <- helper_script()
  out_json <- file.path(out_dir, "federated_checkpoint.json")
  args <- c(
    helper,
    "--samples", samples_csv,
    "--image-root", image_root,
    "--target-col", "label",
    "--path-col", "relative_path",
    "--checkpoint", checkpoint,
    "--output", out_json,
    "--batch-size", as.character(model_params$batch_size),
    "--lr", as.character(model_params$learning_rate),
    "--image-size", as.character(model_params$image_size),
    "--n-classes", as.character(model_params$n_classes),
    "--seed", as.character(seed),
    "--threads", as.character(model_params$threads %||% 2L)
  )
  run_checked(python, args, "Federated checkpoint evaluation failed.",
              timeout = 1800)
  jsonlite::fromJSON(out_json, simplifyVector = FALSE)
}

run_federated <- function(cfg, table_paths, model_spec, rounds,
                          samples_csv, image_root, seed, out_dir) {
  message("\n=== PathMNIST direct image run ===")
  set_image_options(cfg, server_root)
  conns <- connect_tables(cfg, table_paths)
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  recipe <- ds.flower.recipe(
    model = model_spec,
    strategy = ds.flower.strategy.fedavg(),
    target = "label",
    num_rounds = rounds
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

  symbol <- "flower_pathmnist"
  with_ds_errors(ds.flower.nodes.init(conns, data = "D", symbol = symbol))
  on.exit(try(ds.flower.nodes.cleanup(conns, symbol = symbol), silent = TRUE),
          add = TRUE)

  with_ds_errors(
    ds.flower.nodes.prepare(
      conns,
      symbol = symbol,
      target_column = "label",
      feature_columns = NULL,
      run_config = run_config,
      template_name = recipe$model$template
    )
  )
  with_ds_errors(
    ds.flower.nodes.ensure(conns, symbol = symbol,
                           template_name = recipe$model$template)
  )
  run <- with_ds_errors(ds.flower.run.start(recipe, conns = conns, verbose = TRUE))
  with_ds_errors(ds.flower.nodes.cleanup(conns, symbol = symbol))
  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
    error = function(e) NULL
  )
  validate_run(run, post_caps)

  hist <- safe_history(run)
  checkpoint <- file.path(run$output_dir %||% "", "model.pt")
  checkpoint_eval <- NULL
  if (file.exists(checkpoint)) {
    checkpoint_eval <- evaluate_checkpoint(
      samples_csv, image_root, checkpoint, recipe$model$params,
      seed = seed, out_dir = out_dir
    )
  }

  list(
    status = if (identical(as.integer(run$status), 0L)) "ok" else "failed",
    model = recipe$model$name,
    template = recipe$model$template,
    rounds = recipe$num_rounds,
    history = hist,
    final_history_loss = history_metric(hist, "loss"),
    final_history_failures = history_metric(hist, "n_failures"),
    output_dir = run$output_dir %||% NA_character_,
    saved_model = checkpoint,
    saved_model_exists = file.exists(checkpoint),
    checkpoint_eval = checkpoint_eval,
    active_supernodes_after = as.list(vapply(
      post_caps, function(x) x$active_supernodes %||% NA_integer_, integer(1)
    ))
  )
}

cfg <- demo_config("pathmnist_direct_image_resnet", default_rounds = 2L)
n_sites <- length(cfg$urls)
n_per_site <- as.integer(demo_env("DSFLOWER_PATHMNIST_N_PER_SITE", "500"))
image_size <- as.integer(demo_env("DSFLOWER_PATHMNIST_IMAGE_SIZE", "32"))
server_root <- demo_env("DSFLOWER_PATHMNIST_SERVER_ROOT",
                        "/tmp/dsflower_pathmnist_direct_image")
local_root <- demo_env(
  "DSFLOWER_PATHMNIST_LOCAL_ROOT",
  file.path(tempdir(), "dsflower_pathmnist_direct_image")
)
containers <- trimws(strsplit(
  demo_env("DSFLOWER_PATHMNIST_DOCKER_CONTAINERS",
           paste0("opal", seq_len(n_sites), "-rock", collapse = ",")),
  ",", fixed = TRUE
)[[1]])
skip_docker_copy <- demo_bool("DSFLOWER_PATHMNIST_SKIP_DOCKER_COPY", FALSE)

if (!skip_docker_copy && length(containers) != n_sites) {
  stop("DSFLOWER_PATHMNIST_DOCKER_CONTAINERS must have one container per Opal URL.",
       call. = FALSE)
}
if (n_per_site < 500L) {
  stop("PathMNIST direct-image validation needs >=500 rows per site.",
       call. = FALSE)
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(
  demo_env("DSFLOWER_PATHMNIST_OUTPUT_ROOT",
           file.path(getwd(), "dsflower_output", "pathmnist_direct_image")),
  timestamp
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Preparing PathMNIST files under ", local_root)
dataset <- build_pathmnist_dataset(local_root, n_sites, n_per_site, cfg$seed)
samples <- dataset$samples
samples_csv <- file.path(out_dir, "pooled_samples.csv")
utils::write.csv(samples, samples_csv, row.names = FALSE)

if (!skip_docker_copy) {
  if (!nzchar(Sys.which("docker"))) {
    stop("Docker is required for the default local PathMNIST copy.",
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
  site_data <- samples[samples$site == i, c("id", "relative_path",
                                            "source_class", "label"),
                       drop = FALSE]
  table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name, site_data)
  opalr::opal.logout(opal)
}

model_spec <- ds.flower.model.pytorch_resnet18(
  n_classes = 2L,
  learning_rate = as.numeric(demo_env("DSFLOWER_PATHMNIST_LR", "0.001")),
  batch_size = as.integer(demo_env("DSFLOWER_PATHMNIST_BATCH_SIZE", "64")),
  local_epochs = as.integer(demo_env("DSFLOWER_PATHMNIST_LOCAL_EPOCHS", "1")),
  image_size = image_size
)

message("Running centralized PathMNIST ResNet baseline...")
central <- run_central_resnet(
  samples_csv = samples_csv,
  image_root = local_root,
  model_params = model_spec$params,
  rounds = cfg$rounds,
  seed = cfg$seed,
  out_dir = out_dir
)

started_superlink <- FALSE
if (!isTRUE(ds.flower.superlink.status()$running)) {
  ds.flower.superlink.start(detached = FALSE)
  started_superlink <- TRUE
}
on.exit({
  if (started_superlink) try(ds.flower.superlink.stop(), silent = TRUE)
}, add = TRUE)

fed <- run_federated(
  cfg = cfg,
  table_paths = table_paths,
  model_spec = model_spec,
  rounds = cfg$rounds,
  samples_csv = samples_csv,
  image_root = local_root,
  seed = cfg$seed,
  out_dir = out_dir
)

fed_eval <- fed$checkpoint_eval
central_loss <- as.numeric(central$eval$loss %||% NA_real_)
fed_loss <- as.numeric(fed_eval$eval$loss %||% NA_real_)
delta_loss <- fed_loss - central_loss
accepted <- is.finite(central_loss) &&
  is.finite(fed_loss) &&
  fed_loss <= central_loss * 2.5 + 0.25 &&
  as.numeric(fed$final_history_failures %||% 0) <= 0

site_counts <- do.call(rbind, lapply(seq_len(n_sites), function(i) {
  site_data <- samples[samples$site == i, , drop = FALSE]
  data.frame(
    site = paste0("opal", i),
    rows = nrow(site_data),
    label0 = sum(site_data$label == 0L),
    label1 = sum(site_data$label == 1L),
    stringsAsFactors = FALSE
  )
}))

payload <- list(
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
  demo_id = "pathmnist_direct_image_resnet",
  dataset = dataset$metadata,
  citation_key = "yang2023medmnist",
  data_mode = "MedMNIST PathMNIST PNG files + Opal metadata tables",
  n_sites = n_sites,
  n_per_site = n_per_site,
  n_total = nrow(samples),
  site_counts = site_counts,
  image_size = image_size,
  server_root = server_root,
  local_root = normalizePath(local_root, mustWork = FALSE),
  rounds = cfg$rounds,
  model = model_spec$name,
  template = model_spec$template,
  strategy = "FedAvg",
  centralized = central,
  federated = fed,
  federated_comparison = list(
    centralized_loss = central_loss,
    centralized_accuracy = as.numeric(central$eval$accuracy %||% NA_real_),
    federated_checkpoint_loss = fed_loss,
    federated_checkpoint_accuracy = as.numeric(fed_eval$eval$accuracy %||% NA_real_),
    federated_history_loss = as.numeric(fed$final_history_loss %||% NA_real_),
    federated_failures = as.numeric(fed$final_history_failures %||% NA_real_),
    delta_loss = delta_loss,
    delta_accuracy = as.numeric(fed_eval$eval$accuracy %||% NA_real_) -
      as.numeric(central$eval$accuracy %||% NA_real_),
    acceptance_max_loss_ratio = 2.5,
    acceptance_max_loss_margin = 0.25,
    validation_status = if (isTRUE(accepted)) "pass" else "failed"
  )
)

json_path <- file.path(out_dir, "pathmnist_direct_image_resnet_results.json")
jsonlite::write_json(payload, json_path, auto_unbox = TRUE, pretty = TRUE,
                     digits = NA, null = "null")

extdata_dir <- file.path("inst", "extdata")
if (dir.exists(extdata_dir)) {
  file.copy(json_path,
            file.path(extdata_dir, "dsflower_pathmnist_direct_image_results.json"),
            overwrite = TRUE)
}

print(payload$federated_comparison)
cat("Output: ", normalizePath(out_dir), "\n", sep = "")
if (!identical(payload$federated_comparison$validation_status, "pass")) {
  stop("PathMNIST direct-image validation failed.", call. = FALSE)
}
cat("PASS\n")

invisible(payload)
