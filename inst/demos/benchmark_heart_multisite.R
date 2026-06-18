# Real multi-institution UCI Heart Disease federated benchmark.
#
# Unlike benchmark_heart_disease.R (which partitions one Cleveland cohort into
# N synthetic shards), this script loads three REAL institution files and maps
# one institution to one Opal node:
#
#   opal1 <- Cleveland (USA)
#   opal2 <- Hungary (Budapest)
#   opal3 <- VA Long Beach (USA)
#
# Switzerland is excluded on purpose: chol is all zeros, fbs is mostly missing,
# and its outcome is almost entirely positive.
#
# Feature set drops ca/thal/slope, which are missing for every site except
# Cleveland. The remaining 10 features survive complete-case filtering at all
# three institutions.

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

FEATURES <- c("age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak")

INSTITUTIONS <- list(
  list(key = "cleveland", label = "Cleveland (USA)",
       file = "processed.cleveland.data"),
  list(key = "hungary", label = "Hungary (Budapest)",
       file = "processed.hungarian.data"),
  list(key = "va", label = "VA Long Beach (USA)",
       file = "processed.va.data")
)

UCI_BASE <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

load_institution <- function(inst) {
  url <- file.path(UCI_BASE, inst$file)
  tmp <- tempfile(fileext = ".data")
  utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
  cols <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num")
  raw <- utils::read.table(tmp, sep = ",", na.strings = "?", col.names = cols)
  keep <- stats::complete.cases(raw[, c(FEATURES, "num"), drop = FALSE])
  raw <- raw[keep, , drop = FALSE]
  data.frame(
    id = paste0(inst$key, "_", seq_len(nrow(raw))),
    raw[FEATURES],
    outcome = as.integer(raw$num > 0),
    institution = inst$key,
    check.names = FALSE
  )
}

run_multisite <- function() {
  target <- "outcome"
  cfg <- demo_config("uci_heart_multisite", default_rounds = 12L)
  model_name <- demo_env("DSFLOWER_DEMO_MODEL", "sklearn_sgd")
  test_frac <- as.numeric(demo_env("DSFLOWER_DEMO_TEST_FRAC", "0.20"))
  if (length(cfg$urls) < length(INSTITUTIONS)) {
    stop("Need at least ", length(INSTITUTIONS), " Opal URLs; got ", length(cfg$urls))
  }

  # 1. Load each institution, per-institution stratified train/test split.
  train_parts <- list()
  test_parts <- list()
  site_summary <- list()
  for (i in seq_along(INSTITUTIONS)) {
    inst <- INSTITUTIONS[[i]]
    dat <- load_institution(inst)
    for (f in FEATURES) dat[[f]] <- as.numeric(dat[[f]])
    dat[[target]] <- as.integer(dat[[target]])
    split <- stratified_train_test(dat, target = target, test_frac = test_frac, seed = cfg$seed)
    if (nrow(split$train) < 50L) {
      stop(inst$label, " has fewer than 50 training rows after preprocessing.")
    }
    train_parts[[inst$key]] <- split$train
    test_parts[[inst$key]] <- split$test
    site_summary[[inst$key]] <- list(
      label = inst$label,
      n_total = nrow(dat),
      n_train = nrow(split$train),
      n_test = nrow(split$test),
      prevalence = round(mean(dat[[target]]), 4)
    )
    message(sprintf("%-20s total=%-4d train=%-4d test=%-3d prevalence=%.2f",
                    inst$label, nrow(dat), nrow(split$train),
                    nrow(split$test), mean(dat[[target]])))
  }

  train_all <- do.call(rbind, train_parts)
  test_all <- do.call(rbind, test_parts)
  rownames(train_all) <- NULL
  rownames(test_all) <- NULL

  # Keep raw (unstandardized) test rows for human-readable patient examples.
  test_raw <- test_all

  # 2. Standardize using pooled TRAIN statistics (model still trains federated).
  if (demo_bool("DSFLOWER_DEMO_STANDARDIZE", TRUE)) {
    scaled <- standardize_train_test(train_all, test_all, FEATURES)
    train_all <- scaled$train
    test_all <- scaled$test
  }

  # 3. Pooled central baseline (the "if we could pool the data" reference).
  central <- fit_local_glm(train_all, test_all, FEATURES, target)

  # 4. Upload each institution's standardized TRAIN shard to its own Opal.
  table_paths <- character(length(INSTITUTIONS))
  for (i in seq_along(INSTITUTIONS)) {
    inst <- INSTITUTIONS[[i]]
    opal <- opal_login(cfg, i)
    on.exit(try(opalr::opal.logout(opal), silent = TRUE), add = TRUE)
    ensure_project(opal, cfg$project)
    set_privacy_profile(opal, "trusted_internal")
    shard <- train_all[train_all$institution == inst$key, c("id", FEATURES, target), drop = FALSE]
    table_name <- paste0("heart_multisite_", inst$key)
    table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name, shard)
    opalr::opal.logout(opal)
  }

  # 5. Federated training: one institution per node.
  builder <- DSI::newDSLoginBuilder()
  for (i in seq_along(INSTITUTIONS)) {
    builder$append(
      server = INSTITUTIONS[[i]]$key,
      url = cfg$urls[[i]],
      user = cfg$users[[i]],
      password = cfg$passwords[[i]],
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }
  message("Connecting to ", length(INSTITUTIONS), " real institutions...")
  conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  run <- ds.flower.fit(
    conns,
    symbol = "D",
    target = target,
    features = FEATURES,
    model = model_name,
    model_params = list(max_iter = cfg$max_iter),
    strategy = "fedavg",
    privacy = "trusted_internal",
    rounds = cfg$rounds
  )

  # 6. Evaluate federated model on the pooled held-out test set.
  fed_probs <- predict_sklearn_logreg_weights(run$weights, test_all[FEATURES])
  fed_metrics <- binary_metrics(test_all[[target]], fed_probs)

  # 7. Per-institution held-out metrics (federated).
  per_site <- list()
  for (inst in INSTITUTIONS) {
    idx <- which(test_all$institution == inst$key)
    per_site[[inst$key]] <- list(
      label = inst$label,
      n_test = length(idx),
      federated = binary_metrics(test_all[[target]][idx], fed_probs[idx])
    )
  }

  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
    error = function(e) NULL
  )
  validate_run(run, post_caps)

  # 8. Example patient demonstrations (raw clinical values + predicted risk).
  set.seed(cfg$seed)
  pick <- c(
    which(test_raw$institution == "cleveland" & test_raw$outcome == 1L)[1],
    which(test_raw$institution == "hungary" & test_raw$outcome == 0L)[1],
    which(test_raw$institution == "va" & test_raw$outcome == 1L)[1]
  )
  pick <- pick[!is.na(pick)]
  examples <- lapply(pick, function(j) {
    list(
      institution = test_raw$institution[j],
      age = test_raw$age[j], sex = test_raw$sex[j], cp = test_raw$cp[j],
      trestbps = test_raw$trestbps[j], chol = test_raw$chol[j],
      fbs = test_raw$fbs[j], restecg = test_raw$restecg[j],
      thalach = test_raw$thalach[j], exang = test_raw$exang[j],
      oldpeak = test_raw$oldpeak[j],
      predicted_risk = round(fed_probs[j], 4),
      actual_outcome = test_raw$outcome[j]
    )
  })

  # 9. Persist everything for the demo slide.
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(cfg$output_root, paste0("uci_heart_multisite_", timestamp))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  summary <- list(
    demo_id = "uci_heart_multisite",
    dataset_label = "UCI Heart Disease, 3 real institutions",
    institutions = site_summary,
    features = FEATURES,
    n_train = nrow(train_all),
    n_test = nrow(test_all),
    model = model_name,
    strategy = "fedavg",
    privacy = "trusted_internal",
    rounds = cfg$rounds,
    max_iter = cfg$max_iter,
    central_metrics = central$metrics,
    federated_metrics = fed_metrics,
    metric_delta = list(
      auc = fed_metrics$auc - central$metrics$auc,
      accuracy = fed_metrics$accuracy - central$metrics$accuracy,
      log_loss = fed_metrics$log_loss - central$metrics$log_loss
    ),
    per_institution = per_site,
    examples = examples,
    flower_output_dir = run$output_dir %||% NA_character_,
    history = safe_history(run)
  )

  jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                       auto_unbox = TRUE, pretty = TRUE, na = "null")
  saveRDS(list(summary = summary, run = run, central_model = central$model),
          file.path(out_dir, "benchmark.rds"))

  cat("\nUCI Heart Disease, 3 real institutions\n")
  cat("Rounds: ", cfg$rounds, " | local max_iter: ", cfg$max_iter, "\n", sep = "")
  cat("Central AUC: ", sprintf("%.4f", central$metrics$auc),
      " | FL AUC: ", sprintf("%.4f", fed_metrics$auc), "\n", sep = "")
  cat("Central acc: ", sprintf("%.4f", central$metrics$accuracy),
      " | FL acc: ", sprintf("%.4f", fed_metrics$accuracy), "\n", sep = "")
  cat("History (per-round distributed loss):\n")
  print(safe_history(run))
  cat("Output: ", normalizePath(out_dir), "\n", sep = "")
  cat("PASS\n")
  invisible(summary)
}

run_multisite()
