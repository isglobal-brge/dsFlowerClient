#!/usr/bin/env Rscript
# Prepare fixed partitions without inspecting or reporting held-out values.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[1], mustWork = TRUE)
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "cdcgenhlth.R"))
protocol <- jsonlite::fromJSON(file.path(script_dir, "cdcgenhlth_protocol.json"))
run_dir <- file.path(root, "runs", "cdcgenhlth_multiclass")
stopifnot(!dir.exists(run_dir))
dir.create(run_dir, recursive = TRUE)
utc <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
sha <- function(path) digest::digest(file = path, algo = "sha256")
writeLines(utc(), file.path(run_dir, "STARTED"))
path <- file.path(root, "data_cache", "cdc_diabetes_health_indicators.csv")
stopifnot(identical(sha(path), CDCGENHLTH_SHA256),
          identical(CDCGENHLTH_SHA256, protocol$download_sha256))

# Copy of the logistic cdc45k preparation, including column order and CSV format.
full <- utils::read.csv(path)
full$target <- as.integer(full$Diabetes_binary)
full <- full[, c(setdiff(names(full), c("ID", "Diabetes_binary", "target")), "target")]
set.seed(20260819L)
n_target <- 45000
stopifnot(n_target <= nrow(full))
idx <- unlist(lapply(split(seq_len(nrow(full)), full$target), function(ix) {
  take <- round(n_target * length(ix) / nrow(full))
  sample(ix, take)
}), use.names = FALSE)
idx <- sort(idx)
if (length(idx) > n_target) idx <- idx[seq_len(n_target)]
legacy <- full[idx, ]
rownames(legacy) <- NULL
features <- setdiff(names(legacy), "target")
legacy[features] <- lapply(legacy[features], as.numeric)
legacy$target <- as.integer(legacy$target)
legacy_csv <- file.path(run_dir, "legacy_prepared_cdc45k.csv")
utils::write.csv(legacy, legacy_csv, row.names = FALSE)
legacy_sha <- sha(legacy_csv)
stopifnot(identical(legacy_sha, protocol$legacy_logistic_prepared_sha256),
          nrow(legacy) == 45000L,
          identical(CDCGENHLTH_FEATURES, protocol$dataset$features))

df <- legacy[, CDCGENHLTH_FEATURES, drop = FALSE]
df$target <- as.integer(legacy$GenHlth)
rownames(df) <- as.character(idx)
prepared_csv <- file.path(run_dir, "prepared_cdcgenhlth.csv")
utils::write.csv(df, prepared_csv, row.names = FALSE)
seeds <- as.integer(protocol$split$seeds)
splits <- lapply(seeds, function(seed) cdcgenhlth_split(df, seed))
training_path <- file.path(run_dir, "training.rds")
test_path <- file.path(run_dir, "sealed_test.rds")
saveRDS(lapply(splits, function(x) x[c("train", "sites")]), training_path)
saveRDS(lapply(splits, `[[`, "test"), test_path)
preparation <- list(
  schema = "dsflower-cdcgenhlth-preparation-v1", prepared_at_utc = utc(),
  cohort_name = "cdc45k", n_total = nrow(df), cohort_seed = 20260819L,
  split_seeds = seeds, raw_sha256 = CDCGENHLTH_SHA256,
  legacy_prepared_sha256 = legacy_sha,
  cohort_index_sha256 = cdcgenhlth_index_hash(df),
  prepared_sha256 = sha(prepared_csv), training_sha256 = sha(training_path),
  sealed_test_sha256 = sha(test_path),
  rng_kind = RNGkind(), r_version = R.version.string,
  test_policy = "Prepared and sealed; no held-out values or statistics examined or reported.")
jsonlite::write_json(preparation, file.path(run_dir, "preparation.json"),
                     pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
writeLines(utc(), file.path(run_dir, "PREPARATION_COMPLETED"))
cat("Prepared fixed cdc45k training partitions and sealed held-out partitions\n")
