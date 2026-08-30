#!/usr/bin/env Rscript

# Run one campaign cell: contract x dataset x epsilon, `replicates` sequential
# replicates in this single R process (federation built and torn down per
# replicate). Emits one evidence JSON per cell.
#
# Usage:
#   Rscript run_cell.R --contract pytorch_logreg --dataset breast --epsilon 8 \
#     --replicates 3 --rounds 5 --sites 3 --root <work-root> --out <dir>
#
# Requires: R_LIBS pointing at the campaign R library, DSFLOWER_VENV_ROOT and
# DSFLOWER_CLIENT_VENV_ROOT exported, datasets downloaded in <root>/data_cache.

args <- commandArgs(trailingOnly = TRUE)
opt <- list(contract = "pytorch_logreg", dataset = NULL, epsilon = NULL,
            replicates = 3L, rounds = 5L, sites = 3L, root = NULL, out = NULL)
i <- 1L
while (i <= length(args)) {
  key <- sub("^--", "", args[i])
  if (!key %in% names(opt)) stop("Unknown argument: ", args[i], call. = FALSE)
  opt[[key]] <- args[i + 1L]
  i <- i + 2L
}
for (key in c("dataset", "epsilon", "root", "out")) {
  if (is.null(opt[[key]])) stop("--", key, " is required.", call. = FALSE)
}
opt$epsilon <- as.numeric(opt$epsilon)
opt$replicates <- as.integer(opt$replicates)
opt$rounds <- as.integer(opt$rounds)
opt$sites <- as.integer(opt$sites)
wired_contracts <- c("pytorch_logreg", "pytorch_mlp")
if (!opt$contract %in% wired_contracts) {
  stop("Wired contracts: ", paste(wired_contracts, collapse = ", "),
       call. = FALSE)
}

source(file.path(dirname(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])), "campaign_lib.R"))

venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
if (!nzchar(venv_root)) stop("DSFLOWER_VENV_ROOT must be set.", call. = FALSE)
cache_dir <- file.path(opt$root, "data_cache")
runs_dir <- file.path(opt$root, "runs")
dir.create(runs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)

# Frozen after a one-off sanity tuning on breast at epsilon = 8 (see README).
# The MLP reuses the same optimiser settings for comparability and the
# contract-default hidden stack.
model_params <- list(learning_rate = 0.1, batch_size = 32L, local_epochs = 2L)
if (identical(opt$contract, "pytorch_mlp")) {
  model_params$hidden_layers <- c(64L, 32L)
}
delta <- 1e-6
base_seed <- 20260819L

cohort <- campaign_load_cohort(opt$dataset, cache_dir)
df <- cohort$df
features <- cohort$meta$features
cell_id <- sprintf("pilot_%s_%s_eps%g", opt$dataset, opt$contract, opt$epsilon)
cat(sprintf("== cell %s: n=%d, %d features, %d replicates, %d rounds ==\n",
            cell_id, nrow(df), length(features), opt$replicates, opt$rounds))

env_info <- campaign_env_info(venv_root)
per_replicate <- list()
split_info <- NULL

for (r in seq_len(opt$replicates)) {
  seed <- base_seed + r
  split <- campaign_split(df, seed, n_sites = opt$sites)
  bounds <- campaign_bounds(split$train, features)
  if (is.null(split_info)) {
    split_info <- list(
      type = "uniform-stratified",
      test_fraction = 0.2,
      seeds = base_seed + seq_len(opt$replicates),
      n_train = nrow(split$train),
      n_test = nrow(split$test),
      n_per_site = vapply(split$sites, nrow, integer(1))
    )
  }

  central <- if (identical(opt$contract, "pytorch_mlp")) {
    campaign_central_torch_mlp(split$train, split$test, features, model_params)
  } else {
    campaign_central(split$train, split$test, features)
  }
  trivial <- campaign_trivial(split$train, split$test)

  work_dir <- file.path(runs_dir, cell_id, sprintf("rep%d", r))
  fed <- campaign_run_federated(
    site_data = split$sites,
    test = split$test,
    features = features,
    feature_bounds = bounds,
    epsilon = opt$epsilon,
    delta = delta,
    rounds = opt$rounds,
    model_params = model_params,
    work_dir = work_dir,
    venv_root = venv_root,
    contract = opt$contract
  )

  cat(sprintf(
    "rep %d (seed %d): central AUC %.4f | fedDP AUC %.4f acc %.4f brier %.4f | trivial acc %.4f | %.0fs\n",
    r, seed, central$auc, fed$metrics$auc, fed$metrics$acc,
    fed$metrics$brier, trivial$acc, fed$elapsed_s))

  per_replicate[[r]] <- list(
    seed = seed,
    central = central,
    federated_dp = fed$metrics,
    trivial = trivial,
    feature_bounds = bounds,
    history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                   n_rounds = fed$n_rounds_run),
    model_sha256 = fed$model_sha256,
    elapsed_s = round(fed$elapsed_s, 1)
  )
}

msd <- function(metric, branch) {
  values <- vapply(per_replicate, function(rep) rep[[branch]][[metric]],
                   numeric(1))
  list(mean = mean(values), sd = stats::sd(values))
}
deltas <- vapply(per_replicate, function(rep) {
  rep$federated_dp$auc - rep$central$auc
}, numeric(1))

artifact <- list(
  schema = "dsflower-campaign-v1",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  host = env_info$host,
  versions = env_info$versions,
  contract = opt$contract,
  dataset = c(cohort$meta[c("name", "n_total")],
              list(n_train = split_info$n_train,
                   n_test = split_info$n_test,
                   n_per_site = split_info$n_per_site),
              cohort$meta[c("features", "source_url", "sha_of_prepared")]),
  split = split_info[c("type", "test_fraction", "seeds")],
  privacy = list(epsilon = opt$epsilon, delta = delta,
                 unit = "row (default dp_unit policy)",
                 clipping = "dsFlower 0.4.1 defaults",
                 bounds_policy = paste(
                   "feature_bounds declared from the pooled TRAIN side of each",
                   "replicate: observed min/max widened by 10% of the range")),
  rounds = opt$rounds,
  model_params = model_params,
  per_replicate = per_replicate,
  summary = list(
    central_mean_sd = list(
      auc = msd("auc", "central"), acc = msd("acc", "central"),
      brier = msd("brier", "central"), logloss = msd("logloss", "central")),
    federated_mean_sd = list(
      auc = msd("auc", "federated_dp"), acc = msd("acc", "federated_dp"),
      brier = msd("brier", "federated_dp"),
      logloss = msd("logloss", "federated_dp")),
    delta_mean_sd = list(auc = list(mean = mean(deltas), sd = stats::sd(deltas)))
  )
)

out_path <- file.path(opt$out, paste0(cell_id, ".json"))
jsonlite::write_json(artifact, out_path, auto_unbox = TRUE, digits = 6,
                     pretty = TRUE)
cat("WROTE", out_path, "\n")
