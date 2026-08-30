#!/usr/bin/env Rscript

# Capability demonstration cell: federated cross-validation under one
# node-owned differential-privacy contract. Runs `folds` clean federated
# trainings; only the pooled DP out-of-fold metric vector is released
# (80% of the per-job budget split across fold trainings, 20% for the
# single OOF release). Central twin: pooled out-of-fold glm predictions
# over the same fold count on the pooled data.
#
# Usage:
#   Rscript run_cv_demo.R --dataset heart --epsilon 4 --folds 3 \
#     --replicates 3 --rounds 5 --sites 3 --root <work-root> --out <dir>

args <- commandArgs(trailingOnly = TRUE)
opt <- list(contract = "pytorch_logreg", dataset = NULL, epsilon = NULL,
            folds = 3L, replicates = 3L, rounds = 5L, sites = 3L,
            root = NULL, out = NULL)
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
opt$folds <- as.integer(opt$folds)
opt$replicates <- as.integer(opt$replicates)
opt$rounds <- as.integer(opt$rounds)
opt$sites <- as.integer(opt$sites)

source(file.path(dirname(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])), "campaign_lib.R"))

venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
if (!nzchar(venv_root)) stop("DSFLOWER_VENV_ROOT must be set.", call. = FALSE)
cache_dir <- file.path(opt$root, "data_cache")
runs_dir <- file.path(opt$root, "runs")
dir.create(runs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)

model_params <- list(learning_rate = 0.1, batch_size = 32L, local_epochs = 2L)
delta <- 1e-6
base_seed <- 20260819L

# Pooled out-of-fold glm reference on the same data (no privacy, no
# federation): the classical CV twin of the released pooled OOF metric.
central_cv_oof <- function(df, features, folds, seed) {
  set.seed(seed)
  fold_id <- sample(rep_len(seq_len(folds), nrow(df)))
  oof <- rep(NA_real_, nrow(df))
  for (k in seq_len(folds)) {
    train <- df[fold_id != k, c(features, "target")]
    fit <- suppressWarnings(stats::glm(target ~ ., family = stats::binomial(),
                                       data = train))
    oof[fold_id == k] <- suppressWarnings(stats::predict(
      fit, newdata = df[fold_id == k, features, drop = FALSE],
      type = "response"))
  }
  campaign_metrics(df$target, oof)
}

cohort <- campaign_load_cohort(opt$dataset, cache_dir)
df <- cohort$df
features <- cohort$meta$features
cell_id <- sprintf("cvdemo_%s_%s_f%d_eps%g",
                   opt$dataset, opt$contract, opt$folds, opt$epsilon)
cat(sprintf("== cv cell %s: n=%d, %d features, %d folds, %d replicates ==\n",
            cell_id, nrow(df), length(features), opt$folds, opt$replicates))

env_info <- campaign_env_info(venv_root)
per_replicate <- list()

for (r in seq_len(opt$replicates)) {
  seed <- base_seed + r
  split <- campaign_split(df, seed, n_sites = opt$sites)
  # CV consumes the full training pool federated across sites; the campaign
  # test split stays untouched so the cell remains comparable to fit cells.
  bounds <- campaign_bounds(split$train, features)
  central <- central_cv_oof(split$train, features, opt$folds, seed)

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
    contract = opt$contract,
    cv_folds = opt$folds
  )

  # Print only the scalar metrics; the released tree also carries nested
  # pooled curves (roc / precision_recall / calibration / decision_curve).
  scalar_metrics <- Filter(function(v) is.numeric(v) && length(v) == 1L,
                           fed$cv$metrics)
  cat(sprintf("rep %d (seed %d): central OOF AUC %.4f | released DP OOF: %s | %.0fs\n",
              r, seed, central$auc,
              paste(names(scalar_metrics),
                    vapply(scalar_metrics, function(v)
                      sprintf("%.4f", as.numeric(v)[1]), character(1)),
                    sep = "=", collapse = " "),
              fed$elapsed_s))

  per_replicate[[r]] <- list(
    seed = seed,
    central_oof = central,
    federated_dp_oof = fed$cv$metrics,
    cv = fed$cv[c("model", "folds", "n_nodes", "task")],
    elapsed_s = round(fed$elapsed_s, 1)
  )
}

artifact <- list(
  schema = "dsflower-campaign-cv-v1",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  host = env_info$host,
  versions = env_info$versions,
  contract = opt$contract,
  capability = "cross_validation",
  dataset = cohort$meta[c("name", "n_total", "features", "source_url",
                          "sha_of_prepared")],
  privacy = list(
    epsilon = opt$epsilon, delta = delta,
    unit = "row (default dp_unit policy)",
    budget_split = paste(
      "one node-owned per-job contract: 80% divided across the fold",
      "trainings, 20% for the single pooled DP out-of-fold release")),
  folds = opt$folds,
  rounds = opt$rounds,
  model_params = model_params,
  per_replicate = per_replicate
)

out_path <- file.path(opt$out, paste0(cell_id, ".json"))
jsonlite::write_json(artifact, out_path, auto_unbox = TRUE, digits = 6,
                     pretty = TRUE)
cat("WROTE", out_path, "\n")
