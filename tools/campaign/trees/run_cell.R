#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
opt <- list(contract = "random_forest", dataset = NULL, epsilon = NULL,
            replicates = 3L, rounds = 1L, sites = 3L, root = NULL, out = NULL)
stopifnot(length(args) %% 2L == 0L)
for (i in seq.int(1L, length(args), by = 2L)) {
  key <- sub("^--", "", args[i])
  if (!key %in% names(opt)) stop("Unknown argument: ", args[i])
  opt[[key]] <- args[i + 1L]
}
for (key in c("dataset", "epsilon", "root", "out")) stopifnot(!is.null(opt[[key]]))
for (key in c("replicates", "rounds", "sites")) opt[[key]] <- as.integer(opt[[key]])
opt$epsilon <- as.numeric(opt$epsilon)
stopifnot(opt$contract %in% c("random_forest", "extra_trees"),
          opt$dataset %in% c("breast", "cdc9k"), opt$epsilon %in% c(1, 4, 8),
          opt$replicates == 3L, opt$rounds == 1L, opt$sites == 3L)
tools_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(tools_dir, "campaign_lib.R"))
stopifnot(packageVersion("dsFlower") == "0.5.0",
          packageVersion("dsFlowerClient") == "0.5.0")
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
stopifnot(nzchar(venv_root))
cell_id <- sprintf("pilot_%s_%s_eps%g", opt$dataset, opt$contract, opt$epsilon)
runs_dir <- file.path(opt$root, "runs", "trees", cell_id)
out_path <- file.path(opt$out, paste0(cell_id, ".json"))
if (dir.exists(runs_dir) || file.exists(out_path)) stop("Cell already attempted: ", cell_id)
dir.create(runs_dir, recursive = TRUE)
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
write_json <- function(value, path) jsonlite::write_json(
  value, path, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null")
sha <- function(path) digest::digest(file = path, algo = "sha256")
seeds <- 20260819L + seq_len(3L)
model_params <- ds.flower.model(opt$contract)$params
delta <- 1e-6

# Public-schema declarations: no values or summaries from a test split.
public_schema <- function(dataset, features) {
  if (dataset == "breast") {
    lower <- rep(1, length(features)); upper <- rep(10, length(features))
    cuts <- rep(list(seq(1.5, 9.5, 1)), length(features))
  } else {
    lower <- setNames(rep(0, length(features)), features)
    upper <- setNames(rep(1, length(features)), features)
    cuts <- setNames(rep(list(0.5), length(features)), features)
    ranges <- list(BMI = c(0, 100), GenHlth = c(1, 5), MentHlth = c(0, 30),
                   PhysHlth = c(0, 30), Age = c(1, 13), Education = c(1, 6),
                   Income = c(1, 8))
    binary <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex")
    stopifnot(setequal(features, c(binary, names(ranges))))
    for (name in names(ranges)) {
      lower[name] <- ranges[[name]][1]; upper[name] <- ranges[[name]][2]
      cuts[[name]] <- if (name == "BMI") seq(5, 95, 5) else if (
        name %in% c("MentHlth", "PhysHlth")) seq(4.5, 29.5, 5) else {
          seq(lower[name] + 0.5, upper[name] - 0.5, 1)
        }
    }
  }
  list(bounds = list(lower = unname(lower), upper = unname(upper)), cuts = unname(cuts))
}

cache_dir <- file.path(opt$root, "data_cache")
filename <- if (opt$dataset == "breast") "breast-cancer-wisconsin.data" else {
  "cdc_diabetes_health_indicators.csv"
}
expected_sha <- if (opt$dataset == "breast") {
  "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64"
} else "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
stopifnot(identical(sha(file.path(cache_dir, filename)), expected_sha))
cohort <- campaign_load_cohort(opt$dataset, cache_dir)
features <- cohort$meta$features
schema <- public_schema(opt$dataset, features)
stopifnot(nrow(cohort$df) == if (opt$dataset == "breast") 683L else 9000L)
env_info <- campaign_env_info(venv_root)
env_info$host$pod_id <- "2sy2g4pb3xwgqt"
env_info$host$pod_name <- "pod-flower-tabular"
env_info$host$requested_vcpu <- 32L
env_info$host$requested_memory_gb <- 64L
env_info$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(identical(env_info$versions$runner_sha256, dsFlower:::.compute_harness_hash()))
tool_files <- c("run_cell.R", "campaign_lib.R", "central_train.py", "README.md")
tool_hashes <- setNames(lapply(file.path(tools_dir, tool_files), sha), tool_files)
write_json(list(cell_id = cell_id, started_at = format(Sys.time(), "%FT%T%z"),
                options = opt, model_params = model_params,
                tool_sha256 = tool_hashes), file.path(runs_dir, "registration.json"))

per_replicate <- list()
started <- Sys.time()
cat("START", cell_id, "registry defaults:", toJSON(model_params, auto_unbox = TRUE), "\n")
tryCatch({
  for (r in seq_len(3L)) {
    rep_started <- Sys.time()
    split <- campaign_split(cohort$df, seeds[r], n_sites = 3L)
    work_dir <- file.path(runs_dir, paste0("rep", r))
    fed <- campaign_run_federated(
      site_data = split$sites, test = split$test, features = features,
      feature_bounds = schema$bounds, feature_cuts = schema$cuts,
      epsilon = opt$epsilon, delta = delta, rounds = 1L,
      model_params = list(), work_dir = work_dir, venv_root = venv_root,
      contract = opt$contract)
    # Save the single native scoring result before computing any comparator.
    saveRDS(fed, file.path(work_dir, "scored-native.rds"))
    stopifnot(fed$n_clients == 3L, fed$n_failures == 0L, fed$n_rounds_run == 1L,
              isTRUE(fed$cleanup_ok))
    central_dir <- file.path(work_dir, "central")
    dir.create(central_dir)
    train_path <- file.path(central_dir, "train.csv")
    test_path <- file.path(central_dir, "test.csv")
    utils::write.csv(split$train[, c(features, "target")], train_path, row.names = FALSE)
    utils::write.csv(split$test[, features, drop = FALSE], test_path, row.names = FALSE)
    central_schema <- list(features = features, bounds = schema$bounds,
                           cuts = lapply(schema$cuts, as.list), model_params = model_params)
    schema_path <- file.path(central_dir, "schema.json")
    write_json(central_schema, schema_path)
    central_started <- Sys.time()
    python <- file.path(Sys.getenv("DSFLOWER_CLIENT_VENV_ROOT"), "venv", "bin", "python")
    processx::run(python, c(file.path(tools_dir, "central_train.py"),
      "--train", train_path, "--test", test_path, "--schema", schema_path,
      "--out", file.path(central_dir, "probs.csv"),
      "--details", file.path(central_dir, "details.json"),
      "--seed", as.character(seeds[r]), "--epsilon", as.character(opt$epsilon),
      "--contract", opt$contract,
      "--runner", system.file("flower_app", package = "dsFlower")),
      error_on_status = TRUE, echo = TRUE)
    central_details <- jsonlite::fromJSON(file.path(central_dir, "details.json"))
    central <- campaign_metrics(split$test$target,
      utils::read.csv(file.path(central_dir, "probs.csv"))$prob)
    central_elapsed <- as.numeric(difftime(Sys.time(), central_started, units = "secs"))
    trivial <- campaign_metrics(split$test$target, rep(mean(split$train$target), nrow(split$test)))
    gap <- setNames(lapply(names(central), function(m) fed$metrics[[m]] - central[[m]]), names(central))
    per_replicate[[r]] <- list(
      seed = seeds[r], central = central, federated_dp = fed$metrics, trivial = trivial,
      gap = gap, trivial_training_prevalence = mean(split$train$target),
      n_train = nrow(split$train), n_test = nrow(split$test),
      n_per_site = vapply(split$sites, nrow, integer(1)),
      split_row_ids_sha256 = digest::digest(list(train = rownames(split$train),
        test = rownames(split$test), sites = lapply(split$sites, rownames)), algo = "sha256"),
      central_train_csv_sha256 = sha(train_path),
      feature_bounds = schema$bounds, feature_cuts = lapply(schema$cuts, as.list),
      node_privacy = fed$node_privacy, central_configuration = central_details,
      release_metadata = fed$metadata, release_profile = fed$release_profile,
      history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                     n_rounds = fed$n_rounds_run, raw = fed$raw_history),
      model_sha256 = fed$model_sha256, cleanup_ok = fed$cleanup_ok,
      elapsed_s = as.numeric(difftime(Sys.time(), rep_started, units = "secs")),
      federated_elapsed_s = fed$elapsed_s, central_elapsed_s = central_elapsed)
    write_json(per_replicate[[r]], file.path(work_dir, "replicate.json"))
    cat(sprintf("%s rep %d: central AUC %.6f | fed AUC %.6f acc %.6f | trivial acc %.6f | gap %+.6f | %.1fs\n",
                cell_id, r, central$auc, fed$metrics$auc, fed$metrics$acc,
                trivial$acc, gap$auc, per_replicate[[r]]$elapsed_s))
  }
}, error = function(e) {
  write_json(list(schema = "dsflower-campaign-v2", cell_id = cell_id,
    status = "failed", exact_error = conditionMessage(e), contract = opt$contract,
    dataset = opt$dataset, epsilon = opt$epsilon, rounds = 1L,
    tool_sha256 = tool_hashes, completed_replicates = per_replicate,
    elapsed_s = as.numeric(difftime(Sys.time(), started, units = "secs"))),
    file.path(opt$out, paste0("failure_", cell_id, ".json")))
  stop(e)
})

msd <- function(branch) setNames(lapply(c("auc", "acc", "brier", "logloss"), function(metric) {
  values <- vapply(per_replicate, function(rep) rep[[branch]][[metric]], numeric(1))
  list(mean = mean(values), sd = stats::sd(values))
}), c("auc", "acc", "brier", "logloss"))
summary <- list(central_mean_sd = msd("central"), federated_mean_sd = msd("federated_dp"),
                trivial_mean_sd = msd("trivial"), delta_mean_sd = msd("gap"))
floor_pass <- summary$federated_mean_sd$acc$mean >= summary$trivial_mean_sd$acc$mean - 0.02 ||
  summary$federated_mean_sd$auc$mean > 0.6
dataset <- c(cohort$meta, list(
  uci_id = if (opt$dataset == "breast") 15L else 891L,
  full_name = if (opt$dataset == "breast") "Breast Cancer Wisconsin (Original)" else "CDC Diabetes Health Indicators",
  raw_file = filename, download_sha256 = expected_sha,
  citation = if (opt$dataset == "breast") {
    "Wolberg, W. (1990). Breast Cancer Wisconsin (Original). UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z"
  } else "CDC Diabetes Health Indicators (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919",
  cohort_seed = if (opt$dataset == "cdc9k") 20260819L else NULL,
  n_train = per_replicate[[1]]$n_train, n_test = per_replicate[[1]]$n_test,
  n_per_site = per_replicate[[1]]$n_per_site))
artifact <- list(schema = "dsflower-campaign-v2", status = "completed",
  generated_at = format(Sys.time(), "%FT%T%z"), host = env_info$host,
  versions = c(env_info$versions, list(sklearn = central_details$sklearn)),
  release_commits = list(dsFlower = "408f08c539329e2711260050ab40a6567aa4d89e",
                        dsFlowerClient = "50dda000a32ffcbdd039c2b74c909df451392bfb"),
  tool_sha256 = tool_hashes, contract = opt$contract, dataset = dataset,
  split = list(type = "uniform-stratified", test_fraction = 0.2, seeds = seeds,
               site_assignment = "round-robin per class on training rows; unchanged campaign_split"),
  privacy = list(epsilon = opt$epsilon, delta = delta, unit = "row", clipping_norm = 1,
    adjacency = "replace_one", guarantee_scope = "per-training",
    bounds_policy = "Fixed public domain bounds and cuts; frozen before scoring; see trees/README.md",
    randomness = "Node-owned sticky secret RNG; fresh persistent secret files per site/replicate; not seeded by analyst",
    native_clip_semantics = "Unit bounded statistics on public feature/target domains; clipping_norm=1 is the node pin, not a DP-SGD forest mechanism",
    reported_configuration = "See every replicate's node_privacy; native accounting is separately labelled public recomputation"),
  rounds = 1L, rounds_reason = "native-tree schedule", model_params = model_params,
  model_params_overrides = list(), per_replicate = per_replicate, summary = summary,
  diagnostic = list(utility_floor_pass = floor_pass,
    utility_floor = "mean fed accuracy >= mean trivial accuracy - 0.02 OR mean fed AUC > 0.6",
    inherited_from = "vignettes/utility-campaign.Rmd", epsilon8_operating_point = opt$epsilon == 8),
  elapsed_s = as.numeric(difftime(Sys.time(), started, units = "secs")),
  provisioning = list(reused = TRUE, initial_elapsed_s = 97, reprovisioning_elapsed_s = 0))
write_json(artifact, out_path)
cat("WROTE", out_path, "utility floor", if (floor_pass) "PASS" else "FAIL", "\n")
