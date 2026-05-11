`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

demo_env <- function(name, default = NULL) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) default else value
}

demo_flag <- function(name, default = FALSE) {
  value <- demo_env(name, if (isTRUE(default)) "true" else "false")
  tolower(value) %in% c("1", "true", "yes", "y", "on")
}

script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, args, fixed = TRUE, value = TRUE)
  if (length(match)) {
    return(dirname(normalizePath(sub(file_arg, "", match[[1L]], fixed = TRUE),
                                 mustWork = FALSE)))
  }
  getwd()
}

last_history_value <- function(history, field) {
  if (is.null(history)) return(NULL)
  if (is.data.frame(history) && nrow(history) > 0L && field %in% names(history)) {
    return(utils::tail(history[[field]], 1L))
  }
  if (is.list(history) && length(history) > 0L) {
    last <- history[[length(history)]]
    if (is.list(last) && !is.null(last[[field]])) return(last[[field]])
  }
  NULL
}

default_local_workdir <- function() {
  configured <- demo_env("DSFLOWER_IMAGING_LOCAL_WORKDIR", "")
  if (nzchar(configured)) return(configured)
  candidates <- unique(c(
    "/private/tmp/dsimaging_lung1_study",
    "/tmp/dsimaging_lung1_study"
  ))
  found <- candidates[dir.exists(candidates)]
  if (length(found)) normalizePath(found[[1L]], mustWork = FALSE) else ""
}

local_baseline_helper <- function() {
  helper <- system.file("python", "local_resnet_image_baseline.py",
                        package = "dsFlowerClient")
  if (nzchar(helper) && file.exists(helper)) return(helper)
  helper <- file.path(dirname(script_dir()), "python",
                      "local_resnet_image_baseline.py")
  if (file.exists(helper)) normalizePath(helper, mustWork = FALSE) else ""
}

python_has_baseline_deps <- function(python) {
  if (!nzchar(python) || !file.exists(python)) return(FALSE)
  code <- paste(
    "import torch, torchvision, nibabel, pandas, numpy",
    "from PIL import Image",
    sep = "; "
  )
  status <- suppressWarnings(system2(python, c("-c", shQuote(code)),
                                     stdout = FALSE, stderr = FALSE))
  identical(as.integer(status), 0L)
}

local_baseline_python <- function() {
  configured <- demo_env("DSFLOWER_IMAGING_LOCAL_PYTHON", "")
  if (nzchar(configured)) return(configured)
  candidates <- unique(c(
    path.expand("~/.dsflower/venv/bin/python"),
    Sys.which("python3"),
    Sys.which("python"),
    "/opt/homebrew/opt/python@3.11/libexec/bin/python",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3"
  ))
  candidates <- candidates[nzchar(candidates)]
  for (candidate in candidates) {
    if (python_has_baseline_deps(candidate)) return(candidate)
  }
  Sys.which("python3") %||% "python3"
}

run_local_baseline <- function(workdir, target, recipe, out_dir) {
  helper <- local_baseline_helper()
  if (!nzchar(helper)) {
    stop("Could not locate inst/python/local_resnet_image_baseline.py",
         call. = FALSE)
  }
  python <- local_baseline_python()
  baseline_json <- file.path(out_dir, "local_baseline.json")
  baseline_log <- file.path(out_dir, "local_baseline.log")
  args <- c(
    helper,
    "--workdir", normalizePath(workdir, mustWork = FALSE),
    "--target", target,
    "--output", baseline_json,
    "--epochs", as.character(max(1L, recipe$num_rounds *
                                   (recipe$model$params$local_epochs %||% 1L))),
    "--batch-size", as.character(recipe$model$params$batch_size %||% 1L),
    "--lr", as.character(recipe$model$params$learning_rate %||% 5e-4),
    "--seed", demo_env("DSFLOWER_IMAGING_LOCAL_SEED", "4242"),
    "--threads", demo_env("DSFLOWER_IMAGING_LOCAL_THREADS", "2")
  )
  output <- system2(python, args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status") %||% 0L
  writeLines(output, baseline_log)
  if (!identical(as.integer(status), 0L)) {
    stop("Local baseline failed with exit status ", status,
         ". See ", baseline_log, call. = FALSE)
  }
  baseline <- jsonlite::fromJSON(baseline_json, simplifyVector = FALSE)
  baseline$status <- "ok"
  baseline$log <- baseline_log
  baseline
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

suppressPackageStartupMessages({
  library(DSI)
  library(DSOpal)
  library(dsFlowerClient)
  library(jsonlite)
})

urls <- trimws(strsplit(
  demo_env("DSFLOWER_OPAL_URLS",
           "https://localhost:8443,https://localhost:8444,https://localhost:8445"),
  ",", fixed = TRUE
)[[1]])
users <- trimws(strsplit(demo_env("DSFLOWER_OPAL_USERS", "administrator"),
                         ",", fixed = TRUE)[[1]])
passwords <- trimws(strsplit(demo_env("DSFLOWER_OPAL_PASSWORDS", "admin123"),
                             ",", fixed = TRUE)[[1]])
if (length(users) == 1L) users <- rep(users, length(urls))
if (length(passwords) == 1L) passwords <- rep(passwords, length(urls))
if (length(users) != length(urls) || length(passwords) != length(urls)) {
  stop("DSFLOWER_OPAL_USERS and DSFLOWER_OPAL_PASSWORDS must have one value ",
       "or one value per Opal URL.", call. = FALSE)
}

servers <- paste0("opal", seq_along(urls))
resource <- demo_env("DSFLOWER_IMAGING_RESOURCE", "dsdemo.lung1_study")
target <- demo_env("DSFLOWER_IMAGING_TARGET", "os_2yr_alive")
privacy <- demo_env("DSFLOWER_IMAGING_PRIVACY", "sandbox_open")
rounds <- as.integer(demo_env("DSFLOWER_IMAGING_ROUNDS", "1"))
output_root <- demo_env(
  "DSFLOWER_IMAGING_OUTPUT_ROOT",
  file.path(getwd(), "dsflower_output", "direct_dsimaging_images")
)
out_dir <- file.path(output_root, format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
local_workdir <- default_local_workdir()
local_baseline_enabled <- demo_flag("DSFLOWER_IMAGING_LOCAL_BASELINE", TRUE)
require_local_baseline <- demo_flag("DSFLOWER_IMAGING_REQUIRE_LOCAL_BASELINE",
                                    FALSE)

logins <- data.frame(
  server = servers,
  url = urls,
  user = users,
  password = passwords,
  driver = "OpalDriver",
  profile = "default",
  options = "list(ssl_verifyhost=0L, ssl_verifypeer=0L)",
  stringsAsFactors = FALSE
)

message("Logging into Opals...")
conns <- DSI::datashield.login(logins, assign = FALSE)
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

caps <- with_ds_errors(
  DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS"))
)
message("Python env health: ", paste(
  names(caps),
  vapply(caps, function(x) {
    paste(x$python_envs$framework[x$python_envs$healthy], collapse = "/")
  }, character(1)),
  sep = "=", collapse = "; "
))

message("Connecting dsImaging resource ", resource, " to dsFlower...")
flower <- with_ds_errors(ds.flower.connect(conns, resource = resource))
on.exit(try(ds.flower.disconnect(flower), silent = TRUE), add = TRUE)

meta <- with_ds_errors(DSI::datashield.aggregate(
  conns, expr = call("imagingMetadataDS", paste0(flower$symbol, "_img"))
))
assets <- with_ds_errors(DSI::datashield.aggregate(
  conns, expr = call("imagingAssetsDS", paste0(flower$symbol, "_img"))
))
saveRDS(meta, file.path(out_dir, "imaging_metadata.rds"))
saveRDS(assets, file.path(out_dir, "imaging_assets.rds"))

recipe <- ds.flower.recipe(
  model = ds.flower.model.pytorch_resnet18(
    n_classes = 2L,
    learning_rate = as.numeric(demo_env("DSFLOWER_IMAGING_LR", "0.0005")),
    batch_size = as.integer(demo_env("DSFLOWER_IMAGING_BATCH_SIZE", "1")),
    local_epochs = as.integer(demo_env("DSFLOWER_IMAGING_LOCAL_EPOCHS", "1"))
  ),
  strategy = ds.flower.strategy.fedavg(),
  privacy = ds.flower.privacy(privacy),
  target = target,
  num_rounds = rounds
)

message("Running federated ResNet18 over direct dsImaging image assets...")
run <- with_ds_errors(ds.flower.run(flower, recipe, verbose = TRUE))
saveRDS(run, file.path(out_dir, "run.rds"))

local_baseline <- list(
  status = "skipped",
  reason = "disabled",
  enabled = local_baseline_enabled
)
comparison <- NULL
if (isTRUE(local_baseline_enabled)) {
  if (!nzchar(local_workdir) || !dir.exists(local_workdir)) {
    local_baseline <- list(
      status = "skipped",
      reason = "no local dsImaging workdir found",
      enabled = TRUE,
      workdir = local_workdir
    )
    if (isTRUE(require_local_baseline)) {
      stop("Local baseline required but DSFLOWER_IMAGING_LOCAL_WORKDIR was ",
           "not found.", call. = FALSE)
    }
  } else {
    message("Running centralized ResNet18 baseline from ", local_workdir, "...")
    local_baseline <- tryCatch(
      run_local_baseline(local_workdir, target, recipe, out_dir),
      error = function(e) {
        if (isTRUE(require_local_baseline)) stop(e)
        list(status = "failed", reason = conditionMessage(e),
             enabled = TRUE, workdir = local_workdir)
      }
    )
    if (identical(local_baseline$status, "ok")) {
      federated_loss <- as.numeric(last_history_value(run$history, "loss") %||%
                                     NA_real_)
      federated_failures <- as.integer(last_history_value(run$history,
                                                          "n_failures") %||% NA_integer_)
      local_loss <- as.numeric(local_baseline$eval$loss %||% NA_real_)
      comparison <- list(
        note = paste(
          "Same small LUNG1 demo cohort; this is a smoke comparison of the",
          "training/evaluation path, not an external clinical validation."
        ),
        federated = list(
          mode = "federated",
          rounds = recipe$num_rounds,
          clients = length(conns),
          loss = federated_loss,
          n_failures = federated_failures,
          output_dir = run$output_dir %||% NA_character_,
          saved_model = run$saved_path %||% NA_character_
        ),
        centralized = list(
          mode = local_baseline$mode,
          epochs = local_baseline$epochs,
          n_samples = local_baseline$n_samples,
          class_counts = local_baseline$class_counts,
          device = local_baseline$device,
          loss = local_loss,
          accuracy = as.numeric(local_baseline$eval$accuracy %||% NA_real_)
        ),
        delta = list(
          loss_federated_minus_centralized = federated_loss - local_loss
        )
      )
      jsonlite::write_json(
        comparison,
        file.path(out_dir, "local_vs_federated.json"),
        auto_unbox = TRUE,
        pretty = TRUE,
        digits = NA,
        null = "null"
      )
    }
  }
}

post_caps <- with_ds_errors(
  DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS"))
)
saveRDS(post_caps, file.path(out_dir, "post_capabilities.rds"))

summary <- list(
  output_dir = out_dir,
  model_output_dir = run$output_dir %||% NA_character_,
  saved_model = run$saved_path %||% NA_character_,
  resource = resource,
  target = target,
  model = recipe$model$name,
  template = recipe$model$template,
  privacy = recipe$privacy$mode,
  rounds = recipe$num_rounds,
  status = run$status %||% NA_integer_,
  run_id = run$run_id %||% NA_character_,
  history = run$history %||% NULL,
  local_baseline = local_baseline,
  local_vs_federated = comparison,
  metadata_n = as.list(vapply(meta, function(x) x$n_samples %||% NA_integer_,
                              integer(1))),
  active_supernodes_after = as.list(vapply(
    post_caps, function(x) x$active_supernodes %||% NA_integer_, integer(1)
  ))
)
jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA,
                     null = "null")
print(summary)
cat("OUTPUT_DIR=", out_dir, "\n", sep = "")
