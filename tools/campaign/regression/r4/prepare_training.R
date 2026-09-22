# Reuse original sampling functions; split identities before routing model values.
args <- commandArgs(TRUE)
p <- jsonlite::fromJSON(args[1])
cache <- args[3]
out <- args[4]
evidence <- args[5]
original <- new.env(parent = globalenv())
keep <- c("CAMPAIGN_COHORT_SEED", "campaign_load_cohort", "campaign_split")
for (expression in parse(args[2])) {
  if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
      is.symbol(expression[[2]]) && as.character(expression[[2]]) %in% keep) {
    eval(expression, envir = original)
  }
}
stopifnot(original$CAMPAIGN_COHORT_SEED == p$dataset$cohort_seed)
source_path <- file.path(cache, "cdc_diabetes_health_indicators.csv")
stopifnot(file.copy(source_path, file.path(out, basename(source_path))))
cohort <- original$campaign_load_cohort(p$dataset$cohort, out)$df
prepared_sha <- digest::digest(
  file = file.path(out, paste0("prepared_", p$dataset$cohort, ".csv")),
  algo = "sha256")
stopifnot(identical(prepared_sha, p$dataset$logistic_prepared_sha256),
          nrow(cohort) == p$dataset$n_total)
full <- utils::read.csv(source_path)
labels <- as.integer(full$Diabetes_binary)
set.seed(original$CAMPAIGN_COHORT_SEED)
idx <- unlist(lapply(split(seq_len(nrow(full)), labels), function(ix) {
  sample(ix, round(as.numeric(p$dataset$n_total) * length(ix) / nrow(full)))
}), use.names = FALSE)
idx <- sort(idx)
if (length(idx) > p$dataset$n_total) idx <- idx[seq_len(p$dataset$n_total)]
stopifnot(length(idx) == nrow(cohort), identical(labels[idx], cohort$target))
rows <- data.frame(cohort_index = seq_len(nrow(cohort)), source_row = idx,
                   source_id = full$ID[idx], target = cohort$target)
identity <- function(frame) list(cohort_indices = frame$cohort_index,
  source_rows = frame$source_row, uci_ids = frame$source_id)
splits <- list()
for (seed in p$seeds) {
  split <- original$campaign_split(rows, seed, p$sites)
  archived <- jsonlite::fromJSON(file.path(
    evidence, paste0("cdcbmi_split_seed", seed, ".json")), simplifyVector = FALSE)
  same_identity <- function(frame, saved) {
    all(vapply(names(saved), function(key) {
      identical(as.numeric(identity(frame)[[key]]), as.numeric(unlist(saved[[key]])))
    }, logical(1)))
  }
  stopifnot(same_identity(split$train, archived$train),
            same_identity(split$test, archived$test),
            all(vapply(seq_len(p$sites), function(s) {
              same_identity(split$sites[[s]], archived$sites[[s]])
            }, logical(1))))
  folder <- file.path(out, paste0("cdcbmi_seed", seed))
  dir.create(folder)
  columns <- c(p$features, p$target)
  write.csv(cohort[split$train$cohort_index, columns, drop = FALSE],
            file.path(folder, "train.csv"), row.names = FALSE)
  for (s in seq_len(p$sites)) {
    write.csv(cohort[split$sites[[s]]$cohort_index, columns, drop = FALSE],
              file.path(folder, paste0("site", s, ".csv")), row.names = FALSE)
  }
  splits[[as.character(seed)]] <- list(n_train = nrow(split$train),
    n_per_site = vapply(split$sites, nrow, integer(1)), membership_verified = TRUE)
}
jsonlite::write_json(list(cohort_seed = original$CAMPAIGN_COHORT_SEED,
  logistic_prepared_sha256 = prepared_sha, r_version = R.version.string,
  rng_kind = RNGkind(), splits = splits), file.path(out, "preparation.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = NA)
