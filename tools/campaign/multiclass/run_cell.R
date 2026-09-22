#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
opt <- list(root = NULL, out = NULL, epsilon = NULL)
stopifnot(length(args) %% 2L == 0L)
for (i in seq.int(1L, length(args), by = 2L)) {
  key <- sub("^--", "", args[i])
  stopifnot(key %in% names(opt))
  opt[[key]] <- args[i + 1L]
}
stopifnot(all(vapply(opt, Negate(is.null), logical(1))))
opt$epsilon <- as.numeric(opt$epsilon)
stopifnot(opt$epsilon %in% c(1, 4, 8))
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "campaign_lib.R"))
source(file.path(script_dir, "multiclass.R"))
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
stopifnot(nzchar(venv_root))
cell_id <- sprintf("pilot_ctg_pytorch_multiclass_eps%g", opt$epsilon)
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(opt$out, paste0(cell_id, ".json"))
stopifnot(!file.exists(out_path))
cell_dir <- file.path(opt$root, "runs", cell_id)
# Never overwrite or rerun a started cell: failures remain reviewable.
stopifnot(!dir.exists(cell_dir))
dir.create(cell_dir, recursive = TRUE)
writeLines(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), file.path(cell_dir, "STARTED"))
seeds <- 20260819L + 1:3
model_params <- list()  # Every training parameter comes from the v0.5.0 registry.
resolved <- dsFlowerClient:::.dsflower_get_model("pytorch_multiclass")$defaults
resolved <- dsFlowerClient:::.dsflower_complete_training_params(
  resolved, names(dsFlowerClient:::.dsflower_get_model("pytorch_multiclass")$parameter_types))
env_info <- campaign_env_info(venv_root)
stopifnot(env_info$versions$dsflower == "0.5.0", env_info$versions$dsflowerclient == "0.5.0")
env_info$host$pod_id <- "6aq9cxaigfwlby"
env_info$host$pod_name <- "pod-flower-multiclass"
env_info$host$vcpus <- 32L
env_info$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
env_info$versions$dsflower_source_commit <- "408f08c539329e2711260050ab40a6567aa4d89e"
env_info$versions$dsflowerclient_release_commit <- "50dda000a32ffcbdd039c2b74c909df451392bfb"
env_info$versions$nnet <- as.character(packageVersion("nnet"))
script_paths <- file.path(script_dir, c("run_cell.R", "campaign_lib.R", "multiclass.R", "protocol.json"))
script_hashes <- setNames(lapply(script_paths, function(p) digest::digest(file = p, algo = "sha256")), basename(script_paths))
protocol <- jsonlite::fromJSON(file.path(script_dir, "protocol.json"), simplifyVector = FALSE)
df <- ctg_load(file.path(opt$root, "data_cache"))
features <- CTG_FEATURES
per_replicate <- list()
class_sizes <- function(x) setNames(as.list(as.integer(table(factor(x$target, levels = 1:3)))), c("1", "2", "3"))
index_hash <- function(x) digest::digest(paste(rownames(x), collapse = ","), algo = "sha256", serialize = FALSE)
for (r in seq_along(seeds)) {
  start <- Sys.time()
  split <- campaign_split(df, seeds[r], n_sites = 3L)
  bounds <- campaign_bounds(split$train, features)
  central_start <- Sys.time()
  central_fit <- ctg_central_fit(split$train, features, bounds, seeds[r])
  central_fit_s <- as.numeric(difftime(Sys.time(), central_start, units = "secs"))
  # No test summaries or predictions have been examined. Model settings and
  # diagnostic were frozen in protocol.json before any cell was started.
  fed <- campaign_run_federated(
    site_data = split$sites, test = split$test, features = features,
    feature_bounds = bounds, epsilon = opt$epsilon, delta = 1e-6,
    rounds = 5L, model_params = model_params,
    work_dir = file.path(cell_dir, sprintf("rep%d", r)), venv_root = venv_root,
    contract = "pytorch_multiclass", score_function = ctg_score_federated)
  p <- predict(central_fit, ctg_transform(split$test, features, bounds), type = "probs")
  central <- ctg_metrics(split$test$target, p[, c("1", "2", "3"), drop = FALSE])
  central$convergence <- central_fit$convergence
  trivial <- ctg_trivial(split$train, split$test)
  stopifnot(fed$n_clients == 3L, fed$n_failures == 0L, fed$n_rounds_run == 5L,
            fed$cleanup_ok)
  rep <- list(seed = seeds[r],
    split = list(n_train = nrow(split$train), n_test = nrow(split$test),
      n_per_site = vapply(split$sites, nrow, integer(1)),
      class_counts_train = class_sizes(split$train), class_counts_test = class_sizes(split$test),
      class_counts_sites = lapply(split$sites, class_sizes),
      train_index_sha256 = index_hash(split$train), test_index_sha256 = index_hash(split$test),
      site_index_sha256 = lapply(split$sites, index_hash)),
    central = central, federated_dp = fed$metrics, trivial = trivial,
    gap_macro_auc = fed$metrics$macro_auc - central$macro_auc,
    feature_bounds = bounds, node_privacy = fed$node_privacy,
    released_metadata = fed$metadata,
    history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                   n_rounds = fed$n_rounds_run, rounds = fed$history),
    model_sha256 = fed$model_sha256,
    diagnostic = list(accuracy_above_majority = fed$metrics$acc > trivial$acc,
      macro_auc_above_half = fed$metrics$macro_auc > 0.5,
      passed = fed$metrics$acc > trivial$acc && fed$metrics$macro_auc > 0.5),
    wall_clock = list(central_fit_s = central_fit_s, federated_s = fed$elapsed_s,
      total_s = as.numeric(difftime(Sys.time(), start, units = "secs"))),
    cleanup_ok = fed$cleanup_ok)
  per_replicate[[r]] <- rep
  jsonlite::write_json(rep, file.path(cell_dir, sprintf("rep%d.json", r)),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
  cat(sprintf("rep %d: central macro-AUC %.6f | fed %.6f accuracy %.6f | majority %.6f | %.1fs\n",
              r, central$macro_auc, fed$metrics$macro_auc, fed$metrics$acc,
              trivial$acc, rep$wall_clock$total_s))
}
msd <- function(values) list(mean = mean(values), sd = sd(values))
branch_summary <- function(branch) setNames(lapply(c("macro_auc", "acc", "logloss"),
  function(metric) msd(vapply(per_replicate, function(x) x[[branch]][[metric]], numeric(1)))),
  c("macro_auc", "acc", "logloss"))
summary <- list(central_mean_sd = branch_summary("central"),
  federated_mean_sd = branch_summary("federated_dp"),
  trivial_mean_sd = branch_summary("trivial"),
  gap_mean_sd = list(macro_auc = msd(vapply(per_replicate, `[[`, numeric(1), "gap_macro_auc"))),
  diagnostic = list(applicable_epsilon = 8,
    mean_accuracy_above_majority = mean(vapply(per_replicate, function(x) x$federated_dp$acc - x$trivial$acc, numeric(1))) > 0,
    mean_macro_auc_above_half = mean(vapply(per_replicate, function(x) x$federated_dp$macro_auc, numeric(1))) > 0.5,
    all_replicates_pass = all(vapply(per_replicate, function(x) x$diagnostic$passed, logical(1)))))
artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  host = env_info$host, versions = env_info$versions, tooling_sha256 = script_hashes,
  contract = "pytorch_multiclass", dataset = c(protocol$dataset,
    list(n_total = nrow(df), features = features, download_sha256 = CTG_SHA256,
      n_train = per_replicate[[1]]$split$n_train, n_test = per_replicate[[1]]$split$n_test,
      n_per_site = per_replicate[[1]]$split$n_per_site)),
  split = protocol$split, protocol = protocol,
  privacy = list(epsilon = opt$epsilon, delta = 1e-6, clipping_norm = 1,
    unit = "row", adjacency = "replace_one", guarantee_scope = "per-training",
    composition_note = "Epsilon is per node training; these separate runs are not a single composed release guarantee.",
    bounds_policy = protocol$bounds_policy,
    randomness = "Split and central seeds are fixed; DP randomness remains node-owned, cryptographic and sticky. Secrets are never exported."),
  rounds = 5L, model_params_requested = model_params, model_params_resolved = resolved,
  per_replicate = per_replicate, summary = summary)
jsonlite::write_json(artifact, out_path, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null")
cat("WROTE", out_path, "\n")
