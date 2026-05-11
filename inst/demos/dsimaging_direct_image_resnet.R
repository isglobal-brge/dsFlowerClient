`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

demo_env <- function(name, default = NULL) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) default else value
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
  metadata_n = as.list(vapply(meta, function(x) x$n_samples %||% NA_integer_,
                              integer(1))),
  active_supernodes_after = as.list(vapply(
    post_caps, function(x) x$active_supernodes %||% NA_integer_, integer(1)
  ))
)
jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                     auto_unbox = TRUE, pretty = TRUE, null = "null")
print(summary)
cat("OUTPUT_DIR=", out_dir, "\n", sep = "")
