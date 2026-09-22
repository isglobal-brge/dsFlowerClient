#!/usr/bin/env Rscript
# One unscored 0.5.0 reproduction. Preserve public release bytes before cleanup.
source("tools/campaign/trees/campaign_lib.R")
root <- "/workspace/cells"
dest <- file.path(root, "trees-diagnosis-0.5.0")
stopifnot(!dir.exists(dest), packageVersion("dsFlowerClient") == "0.5.0")
dir.create(dest)
trace(".validate_native_tree_ensemble_artifact", where = asNamespace("dsFlowerClient"),
  tracer = quote({
    capture <- "/workspace/cells/trees-diagnosis-0.5.0"
    file.copy(list.files(model_dir, full.names = TRUE), capture, overwrite = TRUE)
    saveRDS(list(meta = meta, task = task, engine = engine), file.path(capture, "validation-input.rds"))
    bytes <- readBin(file.path(model_dir, meta$artifact$file), "raw", n = meta$artifact$size_bytes)
    value <- jsonlite::fromJSON(rawToChar(bytes), simplifyVector = FALSE)
    canonical <- dsFlowerClient:::.native_tree_json(value)
    writeBin(canonical, file.path(capture, "r-canonical.json"))
    n <- min(length(bytes), length(canonical))
    offset <- which(bytes[seq_len(n)] != canonical[seq_len(n)])[1L]
    context <- function(x) rawToChar(x[seq.int(max(1L, offset - 60L), min(length(x), offset + 100L))])
    checks <- list(fields = identical(sort(names(value)), c("aggregation", "contract", "engine", "models", "public_schema_sha256", "task", "version")),
      contract = identical(value$contract, dsFlowerClient:::.native_tree_release_spec(engine)$artifact_contract),
      engine = identical(value$engine, engine), task = identical(value$task, task),
      aggregation = identical(value$aggregation, "mean_prediction"),
      version_integer = is.integer(value$version), version_one = identical(value$version, 1L),
      models = is.list(value$models) && length(value$models) > 0L && all(vapply(value$models, is.list, logical(1))),
      schema_hash = identical(value$public_schema_sha256, meta$public_schema_sha256),
      canonical_bytes = identical(bytes, canonical))
    jsonlite::write_json(list(checks = checks, first_difference_one_based = offset,
      original_context = context(bytes), r_context = context(canonical),
      original_size = length(bytes), r_size = length(canonical),
      original_last_byte = as.integer(tail(bytes, 1)), r_last_byte = as.integer(tail(canonical, 1)),
      key_order = names(value), jsonlite = as.character(packageVersion("jsonlite")),
      r = R.version.string, artifact = meta$artifact), file.path(capture, "comparison.json"),
      auto_unbox = TRUE, pretty = TRUE)
  }), print = FALSE)
cohort <- campaign_load_cohort("breast", file.path(root, "data_cache"))
split <- campaign_split(cohort$df, 20260820L, n_sites = 3L)
campaign_run_federated(split$sites, test = NULL, features = cohort$meta$features,
  feature_bounds = list(lower = rep(1, 9), upper = rep(10, 9)),
  feature_cuts = rep(list(seq(1.5, 9.5, 1)), 9), epsilon = 1, delta = 1e-6,
  rounds = 1L, model_params = list(), work_dir = file.path(dest, "run"),
  venv_root = Sys.getenv("DSFLOWER_VENV_ROOT"), contract = "random_forest",
  score_function = function(...) stop("Diagnosis must not score a holdout"))
