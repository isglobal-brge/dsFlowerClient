required <- c("DSI", "DSOpal", "dsImagingClient", "dsFlowerClient", "jsonlite")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing required demo packages: ", paste(missing, collapse = ", "),
       call. = FALSE)
}

suppressPackageStartupMessages({
  library(DSI)
  library(DSOpal)
  library(dsImagingClient)
  library(dsFlowerClient)
  library(jsonlite)
})

env <- function(name, default = NULL) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) default else value
}

split_env <- function(name, default) {
  trimws(strsplit(env(name, default), ",", fixed = TRUE)[[1]])
}

urls <- split_env(
  "DSFLOWER_OPAL_URLS",
  env("DSFLOWER_DEMO_URLS",
      "https://localhost:8443,https://localhost:8444,https://localhost:8445")
)
users <- split_env("DSFLOWER_OPAL_USERS", env("OPAL_USER", "administrator"))
passwords <- split_env("DSFLOWER_OPAL_PASSWORDS", env("OPAL_PASSWORD", "admin123"))
if (length(users) == 1L) users <- rep(users, length(urls))
if (length(passwords) == 1L) passwords <- rep(passwords, length(urls))
if (length(users) != length(urls) || length(passwords) != length(urls)) {
  stop("DSFLOWER_OPAL_USERS and DSFLOWER_OPAL_PASSWORDS must have one value ",
       "or one value per Opal URL.", call. = FALSE)
}

workdir <- env("LUNG1_WORKDIR", "/tmp/dsimaging_lung1_full")
result_path <- env("LUNG1_RESULT_RDS",
                   file.path(workdir, "datashield_radiomics_result.rds"))
if (!file.exists(result_path)) {
  stop("Radiomics result RDS not found: ", result_path, call. = FALSE)
}

logins <- data.frame(
  server = paste0("opal", seq_along(urls)),
  url = urls,
  user = users,
  password = passwords,
  driver = "OpalDriver",
  options = "list(ssl_verifyhost=0L, ssl_verifypeer=0L)",
  profile = "default",
  stringsAsFactors = FALSE
)

message("Connecting to ", length(urls), " Opal nodes...")
conns <- DSI::datashield.login(logins = logins, assign = FALSE)
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

resource <- paste0(env("LUNG1_OPAL_PROJECT", "dsdemo"), ".",
                   env("LUNG1_OPAL_RESOURCE", "lung1_full_study"))
message("Initializing dsImaging resource: ", resource)
ds.imaging.init(conns, resource, symbol = "img")

radiomics_result <- readRDS(result_path)
for (srv in names(radiomics_result)) {
  item <- radiomics_result[[srv]]
  message("Loading radiomics asset on ", srv, ": ", item$asset_id)
  ds.imaging.radiomics.load_features(
    conns[srv],
    dataset_id = item$dataset_id,
    asset_id = item$asset_id,
    symbol = "rad",
    include_metadata = TRUE,
    syntactic_names = TRUE
  )
}

dims <- tryCatch({
  if (requireNamespace("dsBaseClient", quietly = TRUE)) {
    dsBaseClient::ds.dim("rad", datasources = conns)
  } else {
    NULL
  }
}, error = function(e) NULL)

features <- c(
  "original_firstorder_Energy",
  "original_shape_Compactness1",
  "original_glrlm_RunLengthNonUniformity",
  "wavelet.HLH_glrlm_RunLengthNonUniformity",
  "age",
  "gender_male"
)

rounds <- as.integer(env("DSFLOWER_RADFLOWER_ROUNDS", "2"))
max_iter <- as.integer(env("DSFLOWER_RADFLOWER_MAX_ITER", "100"))
privacy <- env("DSFLOWER_RADFLOWER_PRIVACY", "trusted_internal")

message("Training federated logistic regression on derived radiomics features...")
fit <- ds.flower.fit(
  conns,
  symbol = "rad",
  target = "os_2yr_alive",
  features = features,
  model = "sklearn_logreg",
  model_params = list(max_iter = max_iter),
  strategy = "fedavg",
  privacy = privacy,
  rounds = rounds
)

if (!identical(as.integer(fit$status), 0L)) {
  stop("ds.flower.fit returned non-zero runtime status: ", fit$status,
       call. = FALSE)
}
if (is.null(fit$weights)) {
  stop("No global model weights were returned.", call. = FALSE)
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_root <- env("DSFLOWER_RADFLOWER_OUTPUT_ROOT",
                file.path(getwd(), "dsflower_output", "radiomics_to_flower"))
out_dir <- file.path(out_root, paste0("lung1_radiomics_to_flower_", timestamp))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

history <- if (is.null(fit$history)) data.frame() else fit$history
summary <- list(
  demo_id = "lung1_radiomics_to_flower",
  resource = resource,
  result_path = normalizePath(result_path),
  n_sites = length(conns),
  features = features,
  target = "os_2yr_alive",
  rounds = rounds,
  max_iter = max_iter,
  privacy = privacy,
  status = fit$status,
  run_id = fit$run_id,
  model_id = fit$model_id,
  flower_output_dir = fit$output_dir,
  dimensions = dims,
  history = history
)
jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                     auto_unbox = TRUE, pretty = TRUE, na = "null")
saveRDS(list(summary = summary, fit = fit, radiomics_result = radiomics_result),
        file.path(out_dir, "workflow.rds"))

cat("\nLUNG1 radiomics to dsFlower\n")
if (!is.null(dims)) print(dims)
print(history)
cat("Weights: ", paste(names(fit$weights), collapse = ", "), "\n", sep = "")
cat("Output: ", normalizePath(out_dir), "\n", sep = "")
cat("PASS\n")

invisible(summary)
