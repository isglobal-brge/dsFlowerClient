script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

required <- c("processx")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing required demo packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

ensure_python_module <- function(module, package_spec = module) {
  dsFlowerClient:::.ensure_client_framework("sklearn")
  python <- dsFlowerClient:::.client_python_cmd()
  check <- processx::run(python, c("-c", sprintf("import %s", module)),
                         error_on_status = FALSE)
  if (check$status == 0L) return(invisible(TRUE))

  uv <- dsFlowerClient:::.ensure_client_uv()
  install <- processx::run(
    uv,
    c("pip", "install", "--python", python, "--quiet", package_spec),
    echo_cmd = FALSE,
    echo = TRUE,
    timeout = 900
  )
  if (install$status != 0L) {
    stop("Could not install Python package: ", package_spec, call. = FALSE)
  }
  invisible(TRUE)
}

load_breast_cancer_wisconsin <- function() {
  if (!requireNamespace("mlbench", quietly = TRUE)) {
    stop("The clinical matrix demo requires the mlbench package.", call. = FALSE)
  }
  data("BreastCancer", package = "mlbench")
  raw <- BreastCancer
  raw <- raw[stats::complete.cases(raw), , drop = FALSE]

  feature_cols <- setdiff(names(raw), c("Id", "Class"))
  features <- paste0("bc_", seq_along(feature_cols))

  dataset <- data.frame(
    id = paste0("bcw_", seq_len(nrow(raw))),
    setNames(lapply(raw[feature_cols], function(x) as.numeric(as.character(x))), features),
    outcome = as.integer(raw$Class == "malignant"),
    check.names = FALSE
  )

  list(
    id = "breast_cancer_wisconsin",
    label = "Breast Cancer Wisconsin (Original)",
    domain = "cytology / oncology",
    source = "mlbench::BreastCancer / UCI Breast Cancer Wisconsin (Original)",
    data = dataset,
    features = features
  )
}

load_uci_cleveland <- function() {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
  tmp <- tempfile(fileext = ".data")
  utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
  cols <- c(
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
  )
  raw <- utils::read.table(tmp, sep = ",", na.strings = "?", col.names = cols)
  raw <- raw[stats::complete.cases(raw), , drop = FALSE]
  dataset <- data.frame(
    id = paste0("cleveland_", seq_len(nrow(raw))),
    raw[setdiff(cols, "num")],
    outcome = as.integer(raw$num > 0),
    check.names = FALSE
  )
  list(
    id = "uci_heart_disease",
    label = "UCI Heart Disease processed Cleveland",
    domain = "cardiology",
    source = "UCI Heart Disease processed Cleveland",
    data = dataset,
    features = setdiff(names(dataset), c("id", "outcome"))
  )
}

load_pima_diabetes <- function() {
  if (!requireNamespace("mlbench", quietly = TRUE)) {
    stop("The clinical matrix demo requires the mlbench package.", call. = FALSE)
  }
  data("PimaIndiansDiabetes", package = "mlbench")
  raw <- PimaIndiansDiabetes
  feature_cols <- setdiff(names(raw), "diabetes")
  dataset <- data.frame(
    id = paste0("pima_", seq_len(nrow(raw))),
    raw[feature_cols],
    outcome = as.integer(raw$diabetes == "pos"),
    check.names = FALSE
  )
  list(
    id = "pima_indians_diabetes",
    label = "Pima Indians Diabetes",
    domain = "endocrinology / diabetes",
    source = "mlbench::PimaIndiansDiabetes / UCI Pima Indians Diabetes",
    data = dataset,
    features = feature_cols
  )
}

load_cdc_diabetes <- function(limit = as.integer(demo_env("DSFLOWER_CDC_DIABETES_LIMIT", 9000L)),
                              seed = as.integer(demo_env("DSFLOWER_CDC_DIABETES_SEED", 4242L))) {
  ensure_python_module("ucimlrepo", "ucimlrepo>=0.0.7")
  python <- dsFlowerClient:::.client_python_cmd()
  out <- tempfile(fileext = ".csv")
  code <- paste(
    "import json, numpy as np, pandas as pd",
    "from ucimlrepo import fetch_ucirepo",
    sprintf("out_path = %s", jsonlite::toJSON(out, auto_unbox = TRUE)),
    sprintf("limit = int(%s)", jsonlite::toJSON(limit, auto_unbox = TRUE)),
    sprintf("seed = int(%s)", jsonlite::toJSON(seed, auto_unbox = TRUE)),
    "ds = fetch_ucirepo(id=891)",
    "X = ds.data.features.copy()",
    "y = ds.data.targets.iloc[:, 0].astype(int).rename('outcome')",
    "df = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)",
    "rng = np.random.default_rng(seed)",
    "idx = []",
    "for cls in sorted(df['outcome'].unique()):",
    "    cls_idx = np.where(df['outcome'].to_numpy() == cls)[0]",
    "    rng.shuffle(cls_idx)",
    "    take = max(1, int(round(limit * len(cls_idx) / len(df))))",
    "    idx.extend(cls_idx[:take].tolist())",
    "idx = np.array(idx[:limit])",
    "rng.shuffle(idx)",
    "df = df.iloc[idx].reset_index(drop=True)",
    "df.insert(0, 'id', [f'cdc_diabetes_{i+1}' for i in range(len(df))])",
    "df.columns = ['id'] + [f'cdc_{i:02d}' for i in range(1, len(df.columns)-1)] + ['outcome']",
    "df.to_csv(out_path, index=False)",
    sep = "\n"
  )
  run <- processx::run(python, c("-c", code), echo = TRUE,
                       error_on_status = FALSE, timeout = 900)
  if (run$status != 0L) {
    stop("Could not build CDC Diabetes Health Indicators dataset:\n",
         run$stderr, call. = FALSE)
  }
  dataset <- utils::read.csv(out, check.names = FALSE)
  list(
    id = "cdc_diabetes_health_indicators",
    label = "CDC Diabetes Health Indicators",
    domain = "population health / diabetes",
    source = "UCI CDC Diabetes Health Indicators, stratified sample",
    data = dataset,
    features = setdiff(names(dataset), c("id", "outcome"))
  )
}

clinical_model_specs <- function() {
  list(
    list(
      id = "sklearn_logreg",
      label = "Logistic regression",
      family = "linear classifier",
      validation_datasets = c(
        "breast_cancer_wisconsin",
        "uci_heart_disease",
        "pima_indians_diabetes",
        "cdc_diabetes_health_indicators"
      ),
      clinical_validation_datasets = c(
        "breast_cancer_wisconsin",
        "pima_indians_diabetes",
        "cdc_diabetes_health_indicators"
      ),
      min_rows_per_site = 50L,
      model = "sklearn_logreg",
      model_params = list(max_iter = as.integer(demo_env("DSFLOWER_CLINICAL_LOGREG_MAX_ITER", 200L))),
      rounds = as.integer(demo_env("DSFLOWER_CLINICAL_LOGREG_ROUNDS", 2L)),
      central = "sklearn_logreg"
    ),
    list(
      id = "xgboost_histogram",
      label = "Histogram XGBoost",
      family = "tree ensemble",
      validation_datasets = c("breast_cancer_wisconsin"),
      clinical_validation_datasets = c("cdc_diabetes_health_indicators"),
      min_rows_per_site = 100L,
      model = "xgboost",
      model_params = list(
        n_trees = as.integer(demo_env("DSFLOWER_CLINICAL_XGB_TREES", 5L)),
        max_depth = as.integer(demo_env("DSFLOWER_CLINICAL_XGB_DEPTH", 1L)),
        eta = as.numeric(demo_env("DSFLOWER_CLINICAL_XGB_ETA", 0.25)),
        reg_lambda = as.numeric(demo_env("DSFLOWER_CLINICAL_XGB_LAMBDA", 1.0)),
        n_bins = as.integer(demo_env("DSFLOWER_CLINICAL_XGB_BINS", 16L)),
        objective = "binary:logistic"
      ),
      rounds = 1L,
      central = "xgboost"
    ),
    list(
      id = "sklearn_ridge",
      label = "Ridge classifier",
      family = "linear classifier",
      validation_datasets = c("cdc_diabetes_health_indicators"),
      clinical_validation_datasets = c("cdc_diabetes_health_indicators"),
      min_rows_per_site = 50L,
      model = "sklearn_ridge",
      model_params = list(
        alpha = as.numeric(demo_env("DSFLOWER_CLINICAL_RIDGE_ALPHA", 1.0))
      ),
      rounds = as.integer(demo_env("DSFLOWER_CLINICAL_RIDGE_ROUNDS", 2L)),
      central = "sklearn_ridge"
    ),
    list(
      id = "sklearn_sgd",
      label = "SGD logistic classifier",
      family = "linear classifier",
      validation_datasets = c("cdc_diabetes_health_indicators"),
      clinical_validation_datasets = c("cdc_diabetes_health_indicators"),
      min_rows_per_site = 100L,
      model = "sklearn_sgd",
      model_params = list(
        loss = "log_loss",
        alpha = as.numeric(demo_env("DSFLOWER_CLINICAL_SGD_ALPHA", 1e-4)),
        max_iter = as.integer(demo_env("DSFLOWER_CLINICAL_SGD_MAX_ITER", 1000L))
      ),
      rounds = as.integer(demo_env("DSFLOWER_CLINICAL_SGD_ROUNDS", 2L)),
      central = "sklearn_sgd"
    ),
    list(
      id = "pytorch_logreg",
      label = "PyTorch logistic regression",
      family = "neural network",
      validation_datasets = c("cdc_diabetes_health_indicators"),
      clinical_validation_datasets = c("cdc_diabetes_health_indicators"),
      min_rows_per_site = 250L,
      model = "pytorch_logreg",
      model_params = list(
        learning_rate = as.numeric(demo_env("DSFLOWER_CLINICAL_TORCH_LOGREG_LR", 0.05)),
        batch_size = as.integer(demo_env("DSFLOWER_CLINICAL_TORCH_LOGREG_BATCH", 128L)),
        local_epochs = as.integer(demo_env("DSFLOWER_CLINICAL_TORCH_LOGREG_LOCAL_EPOCHS", 1L))
      ),
      rounds = as.integer(demo_env("DSFLOWER_CLINICAL_TORCH_LOGREG_ROUNDS", 10L)),
      central = "pytorch_logreg"
    ),
    list(
      id = "pytorch_mlp",
      label = "PyTorch MLP",
      family = "neural network",
      validation_datasets = c("cdc_diabetes_health_indicators"),
      min_rows_per_site = 250L,
      model = "pytorch_mlp",
      model_params = list(
        hidden_layers = demo_env("DSFLOWER_CLINICAL_MLP_HIDDEN", "32,16"),
        learning_rate = as.numeric(demo_env("DSFLOWER_CLINICAL_MLP_LR", 0.005)),
        batch_size = as.integer(demo_env("DSFLOWER_CLINICAL_MLP_BATCH", 64L)),
        local_epochs = as.integer(demo_env("DSFLOWER_CLINICAL_MLP_LOCAL_EPOCHS", 1L))
      ),
      rounds = as.integer(demo_env("DSFLOWER_CLINICAL_MLP_ROUNDS", 6L)),
      central = "pytorch_mlp"
    )
  )
}

filter_by_env <- function(items, env_name, id_field = "id") {
  raw <- demo_env(env_name, "")
  if (!nzchar(raw)) return(items)
  keep <- trimws(strsplit(raw, ",", fixed = TRUE)[[1]])
  filtered <- Filter(function(x) x[[id_field]] %in% keep, items)
  if (!length(filtered)) {
    stop("No entries selected by ", env_name, "=", raw, call. = FALSE)
  }
  filtered
}

profile_min_rows <- function(spec, profile) {
  family <- spec$family
  if (family %in% c("linear classifier")) {
    switch(profile,
      sandbox_open = 3L,
      trusted_internal = 50L,
      consortium_internal = 50L,
      clinical_default = 100L,
      clinical_hardened = 200L,
      clinical_update_noise = Inf,
      high_sensitivity_dp = Inf,
      spec$min_rows_per_site
    )
  } else if (family %in% c("tree ensemble")) {
    switch(profile,
      sandbox_open = 3L,
      trusted_internal = 100L,
      consortium_internal = 100L,
      clinical_default = 200L,
      clinical_hardened = 300L,
      clinical_update_noise = 500L,
      high_sensitivity_dp = Inf,
      spec$min_rows_per_site
    )
  } else if (family %in% c("neural network")) {
    switch(profile,
      sandbox_open = 3L,
      trusted_internal = 250L,
      consortium_internal = 250L,
      clinical_default = 500L,
      clinical_hardened = 750L,
      clinical_update_noise = 500L,
      high_sensitivity_dp = 1000L,
      spec$min_rows_per_site
    )
  } else {
    spec$min_rows_per_site
  }
}

specs_for_dataset <- function(specs, dataset_id) {
  Filter(function(spec) {
    scope <- spec$validation_datasets %||% character(0)
    !length(scope) || dataset_id %in% scope
  }, specs)
}

specs_for_dataset_profile <- function(specs, dataset_id, profile) {
  clinical_profiles <- c(
    "clinical_default", "clinical_hardened",
    "clinical_update_noise", "high_sensitivity_dp"
  )
  Filter(function(spec) {
    scope <- spec$validation_datasets %||% character(0)
    if (profile %in% clinical_profiles) {
      scope <- spec$clinical_validation_datasets %||% scope
    }
    !length(scope) || dataset_id %in% scope
  }, specs)
}

demo_privacy_spec <- function(profile) {
  if (profile %in% c("clinical_update_noise", "high_sensitivity_dp")) {
    return(ds.flower.privacy(
      profile,
      epsilon = as.numeric(demo_env("DSFLOWER_CLINICAL_DP_EPSILON", 1.0)),
      delta = as.numeric(demo_env("DSFLOWER_CLINICAL_DP_DELTA", 1e-5)),
      clipping_norm = as.numeric(demo_env("DSFLOWER_CLINICAL_DP_CLIP", 1.0))
    ))
  }
  ds.flower.privacy(profile)
}

privacy_summary <- function(privacy) {
  c(list(mode = privacy$mode), privacy$params %||% list())
}

python_classifier_metrics <- function(train, test, features, target, spec, seed = 4242L) {
  framework <- switch(spec$central,
    sklearn_logreg = "sklearn",
    sklearn_ridge = "sklearn",
    sklearn_sgd = "sklearn",
    xgboost = "xgboost",
    pytorch_logreg = "pytorch",
    pytorch_mlp = "pytorch",
    "sklearn"
  )
  dsFlowerClient:::.ensure_client_framework("sklearn")
  if (!identical(framework, "sklearn")) dsFlowerClient:::.ensure_client_framework(framework)

  train_csv <- tempfile(fileext = ".csv")
  test_csv <- tempfile(fileext = ".csv")
  out_json <- tempfile(fileext = ".json")
  utils::write.csv(train[, c(features, target), drop = FALSE], train_csv, row.names = FALSE)
  utils::write.csv(test[, c(features, target), drop = FALSE], test_csv, row.names = FALSE)

  payload <- list(
    train = train_csv,
    test = test_csv,
    output = out_json,
    features = features,
    target = target,
    model = spec$central,
    params = spec$model_params,
    rounds = spec$rounds,
    seed = seed
  )
  payload_json <- tempfile(fileext = ".json")
  jsonlite::write_json(payload, payload_json, auto_unbox = TRUE, pretty = TRUE)

  code <- paste(
    "import json, math, random",
    "import numpy as np",
    "import pandas as pd",
    "from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss",
    "cfg = json.load(open(r'''%s'''))",
    "random.seed(int(cfg['seed']))",
    "np.random.seed(int(cfg['seed']))",
    "train = pd.read_csv(cfg['train'])",
    "test = pd.read_csv(cfg['test'])",
    "features = cfg['features']",
    "target = cfg['target']",
    "X_train = train[features].to_numpy(dtype='float32')",
    "y_train = train[target].to_numpy(dtype='int64')",
    "X_test = test[features].to_numpy(dtype='float32')",
    "y_test = test[target].to_numpy(dtype='int64')",
    "model = cfg['model']",
    "params = cfg.get('params', {})",
    "if model == 'sklearn_logreg':",
    "    from sklearn.linear_model import LogisticRegression",
    "    clf = LogisticRegression(max_iter=int(params.get('max_iter', 200)), solver='lbfgs')",
    "    clf.fit(X_train, y_train)",
    "    prob = clf.predict_proba(X_test)[:, 1]",
    "elif model == 'sklearn_ridge':",
    "    from sklearn.linear_model import RidgeClassifier",
    "    clf = RidgeClassifier(alpha=float(params.get('alpha', 1.0)))",
    "    clf.fit(X_train, y_train)",
    "    score = clf.decision_function(X_test)",
    "    prob = 1.0 / (1.0 + np.exp(-score))",
    "elif model == 'sklearn_sgd':",
    "    from sklearn.linear_model import SGDClassifier",
    "    clf = SGDClassifier(",
    "        loss=params.get('loss', 'log_loss'),",
    "        penalty=params.get('penalty', 'l2'),",
    "        alpha=float(params.get('alpha', 1e-4)),",
    "        max_iter=int(params.get('max_iter', 1000)),",
    "        tol=1e-4, random_state=int(cfg['seed']))",
    "    clf.fit(X_train, y_train)",
    "    if hasattr(clf, 'predict_proba'):",
    "        prob = clf.predict_proba(X_test)[:, 1]",
    "    else:",
    "        score = clf.decision_function(X_test)",
    "        prob = 1.0 / (1.0 + np.exp(-score))",
    "elif model == 'xgboost':",
    "    import xgboost as xgb",
    "    clf = xgb.XGBClassifier(",
    "        n_estimators=int(params.get('n_trees', 4)),",
    "        max_depth=int(params.get('max_depth', 2)),",
    "        learning_rate=float(params.get('eta', 0.25)),",
    "        reg_lambda=float(params.get('reg_lambda', 1.0)),",
    "        objective=params.get('objective', 'binary:logistic'),",
    "        eval_metric='logloss', subsample=1.0, colsample_bytree=1.0,",
    "        random_state=int(cfg['seed']), n_jobs=1, verbosity=0)",
    "    clf.fit(X_train, y_train)",
    "    prob = clf.predict_proba(X_test)[:, 1]",
    "elif model in ('pytorch_logreg', 'pytorch_mlp'):",
    "    import torch",
    "    torch.manual_seed(int(cfg['seed']))",
    "    layers = []",
    "    prev = X_train.shape[1]",
    "    if model == 'pytorch_mlp':",
    "        hidden_raw = params.get('hidden_layers', '32,16')",
    "        if isinstance(hidden_raw, str):",
    "            hidden = [int(x) for x in hidden_raw.split(',') if str(x).strip()]",
    "        else:",
    "            hidden = [int(x) for x in hidden_raw]",
    "        for h in hidden:",
    "            layers.append(torch.nn.Linear(prev, h))",
    "            layers.append(torch.nn.ReLU())",
    "            prev = h",
    "    layers.append(torch.nn.Linear(prev, 1))",
    "    net = torch.nn.Sequential(*layers)",
    "    opt = torch.optim.Adam(net.parameters(), lr=float(params.get('learning_rate', 0.005)))",
    "    loss_fn = torch.nn.BCEWithLogitsLoss()",
    "    X = torch.tensor(X_train, dtype=torch.float32)",
    "    y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)",
    "    batch = int(params.get('batch_size', 64))",
    "    epochs = int(params.get('central_epochs', max(6, int(cfg.get('rounds', 6)))))",
    "    gen = torch.Generator().manual_seed(int(cfg['seed']))",
    "    ds = torch.utils.data.TensorDataset(X, y)",
    "    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)",
    "    net.train()",
    "    for _ in range(epochs):",
    "        for xb, yb in loader:",
    "            opt.zero_grad()",
    "            loss = loss_fn(net(xb), yb)",
    "            loss.backward()",
    "            opt.step()",
    "    net.eval()",
    "    with torch.no_grad():",
    "        logits = net(torch.tensor(X_test, dtype=torch.float32)).reshape(-1)",
    "        prob = torch.sigmoid(logits).numpy()",
    "else:",
    "    raise ValueError(f'unknown model {model}')",
    "prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)",
    "metrics = {",
    "    'auc': float(roc_auc_score(y_test, prob)),",
    "    'accuracy': float(accuracy_score(y_test, (prob >= 0.5).astype(int))),",
    "    'log_loss': float(log_loss(y_test, prob, labels=[0, 1])),",
    "    'brier': float(brier_score_loss(y_test, prob)),",
    "}",
    "json.dump({'metrics': metrics, 'probabilities': prob.tolist()}, open(cfg['output'], 'w'))",
    sep = "\n"
  )
  code <- sprintf(code, payload_json)

  python <- dsFlowerClient:::.client_python_cmd()
  run <- processx::run(python, c("-c", code), env = dsFlowerClient:::.client_venv_env(),
                       error_on_status = FALSE, timeout = 1200)
  if (run$status != 0L) {
    stop("Central Python baseline failed for ", spec$id, ":\n", run$stderr, call. = FALSE)
  }
  jsonlite::fromJSON(out_json, simplifyVector = FALSE)
}

predict_dsflower_xgboost_artifact <- function(output_dir, x) {
  model_path <- file.path(output_dir, "global_model.json")
  if (!file.exists(model_path)) {
    stop("XGBoost global_model.json not found in ", output_dir, call. = FALSE)
  }
  model <- jsonlite::fromJSON(model_path, simplifyVector = FALSE)
  if (!identical(model$model_type, "xgboost")) {
    stop("The selected artifact is not a dsFlower XGBoost model.", call. = FALSE)
  }

  x <- as.matrix(x)
  raw <- numeric(nrow(x))
  learning_rate <- as.numeric(model$learning_rate %||% 0.3)

  for (tree in model$trees) {
    splits <- tree$splits %||% list()
    leaves <- tree$leaves %||% list()
    for (i in seq_len(nrow(x))) {
      node <- "0"
      repeat {
        split <- splits[[node]]
        if (is.null(split)) break
        feature_idx <- as.integer(split$feature) + 1L
        threshold <- as.numeric(split$threshold)
        node <- if (x[i, feature_idx] <= threshold) {
          as.character(split$left)
        } else {
          as.character(split$right)
        }
      }
      raw[[i]] <- raw[[i]] + learning_rate * as.numeric(leaves[[node]] %||% 0)
    }
  }
  as.numeric(1 / (1 + exp(-raw)))
}

predict_federated_model <- function(run, spec, test, features) {
  if (spec$id %in% c("sklearn_logreg", "sklearn_ridge", "sklearn_sgd")) {
    return(predict_sklearn_logreg_weights(run$weights, test[features]))
  }
  if (identical(spec$id, "xgboost_histogram")) {
    return(predict_dsflower_xgboost_artifact(run$output_dir, test[features]))
  }
  if (spec$id %in% c("pytorch_logreg", "pytorch_mlp")) {
    return(as.numeric(ds.flower.predict(run, test[features], type = "prob")))
  }
  stop("No prediction adapter for model spec: ", spec$id, call. = FALSE)
}

run_clinical_dataset <- function(dataset, specs, cfg, target = "outcome",
                                 test_frac = 0.25) {
  data <- prepare_for_benchmark(dataset$data, dataset$features, target = target)
  split <- stratified_train_test(data, target = target, test_frac = test_frac, seed = cfg$seed)
  scaled <- standardize_train_test(split$train, split$test, dataset$features)
  train <- scaled$train
  test <- scaled$test

  n_sites <- length(cfg$urls)
  train$site <- make_stratified_site_labels(train[[target]], n_sites = n_sites, seed = cfg$seed)
  site_counts <- as.integer(table(factor(train$site, levels = seq_len(n_sites))))
  names(site_counts) <- paste0("opal", seq_len(n_sites))
  site_tables <- split(train, train$site)

  table_paths <- character(n_sites)
  for (i in seq_len(n_sites)) {
    opal <- opal_login(cfg, i)
    ensure_project(opal, cfg$project)
    set_privacy_profile(
      opal,
      cfg$profile,
      ledger_namespace = privacy_ledger_namespace_for_run(
        cfg, dataset_id = dataset$id, model_id = "dataset", privacy = cfg$privacy),
      max_epsilon = cfg$privacy_max_epsilon,
      max_delta = cfg$privacy_max_delta
    )
    table_name <- paste0(cfg$table_prefix, "_", dataset$id, "_site", i)
    table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name,
                                          site_tables[[as.character(i)]])
    opalr::opal.logout(opal)
  }

  builder <- DSI::newDSLoginBuilder()
  for (i in seq_len(n_sites)) {
    builder$append(
      server = paste0("opal", i),
      url = cfg$urls[[i]],
      user = cfg$users[[i]],
      password = cfg$passwords[[i]],
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }

  message("Connecting to ", n_sites, " Opal nodes for ", dataset$label, "...")
  conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  records <- list()
  dataset_specs <- specs_for_dataset_profile(specs, dataset$id, cfg$profile)
  for (spec in dataset_specs) {
    min_required <- profile_min_rows(spec, cfg$profile)
    if (!is.finite(min_required) || min(site_counts) < min_required) {
      records[[spec$id]] <- list(
        dataset_id = dataset$id,
        dataset_label = dataset$label,
        domain = dataset$domain,
        source = dataset$source,
        model_id = spec$id,
        model_label = spec$label,
        family = spec$family,
        status = "not_run_policy_min_rows",
        min_rows_per_site = min_required,
        smallest_site_train_n = min(site_counts),
        privacy_profile = cfg$profile,
        privacy_parameters = privacy_summary(cfg$privacy),
        note = paste0(
          "The active privacy profile requires a larger per-site training ",
          "set for this model family."
        )
      )
      next
    }

    message("Running ", spec$label, " on ", dataset$label, "...")
    central <- python_classifier_metrics(train, test, dataset$features, target, spec, seed = cfg$seed)
    strategy_spec <- ds.flower.strategy.fedavg(fraction_fit = 1.0, fraction_evaluate = 1.0)
    strategy_spec$params$min_fit_clients <- n_sites
    strategy_spec$params$min_available_clients <- n_sites
    ledger_namespace <- set_run_privacy_options(cfg, dataset$id, spec$id)

    run <- ds.flower.fit(
      conns,
      symbol = "D",
      target = target,
      features = dataset$features,
      model = spec$model,
      model_params = spec$model_params,
      strategy = strategy_spec,
      privacy = cfg$privacy,
      rounds = spec$rounds,
      verbose = identical(toupper(demo_env("DSFLOWER_CLINICAL_VERBOSE", "FALSE")), "TRUE")
    )
    post_caps <- tryCatch(
      DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
      error = function(e) NULL
    )
    validate_run(run, post_caps)
    fed_probs <- predict_federated_model(run, spec, test, dataset$features)
    fed_metrics <- binary_metrics(test[[target]], fed_probs)
    central_metrics <- central$metrics

    records[[spec$id]] <- list(
      dataset_id = dataset$id,
      dataset_label = dataset$label,
      domain = dataset$domain,
      source = dataset$source,
      model_id = spec$id,
      model_label = spec$label,
      family = spec$family,
      status = "pass",
      n_total = nrow(data),
      n_train = nrow(train),
      n_test = nrow(test),
      n_features = length(dataset$features),
      site_train_n = as.list(site_counts),
      privacy_profile = cfg$profile,
      privacy_parameters = privacy_summary(cfg$privacy),
      privacy_ledger_namespace = ledger_namespace,
      rounds = spec$rounds,
      model_params = spec$model_params,
      central_metrics = central_metrics,
      federated_metrics = fed_metrics,
      metric_delta = list(
        auc = fed_metrics$auc - central_metrics$auc,
        accuracy = fed_metrics$accuracy - central_metrics$accuracy,
        log_loss = fed_metrics$log_loss - central_metrics$log_loss,
        brier = fed_metrics$brier - central_metrics$brier
      ),
      flower_output_dir = run$output_dir %||% NA_character_,
      history = safe_history(run)
    )

    hist <- safe_history(run)
    n_failures <- if ("n_failures" %in% names(hist)) {
      sum(as.integer(hist$n_failures), na.rm = TRUE)
    } else {
      0L
    }
    cat(
      dataset$label, " | ", spec$label,
      " | central AUC=", sprintf("%.4f", central_metrics$auc),
      " | federated AUC=", sprintf("%.4f", fed_metrics$auc),
      " | failures=", n_failures,
      "\n", sep = ""
    )
  }

  list(
    dataset_id = dataset$id,
    dataset_label = dataset$label,
    domain = dataset$domain,
    source = dataset$source,
    n_total = nrow(data),
    n_train = nrow(train),
    n_test = nrow(test),
    n_features = length(dataset$features),
    site_train_n = as.list(site_counts),
    train_prevalence = mean(train[[target]]),
    test_prevalence = mean(test[[target]]),
    results = records
  )
}

run_clinical_algorithm_matrix <- function() {
  cfg <- demo_config("clinical_algorithm_matrix", default_rounds = 2L)
  cfg$privacy <- demo_privacy_spec(cfg$profile)
  cfg$profile <- cfg$privacy$mode
  cfg$table_prefix <- demo_env("DSFLOWER_CLINICAL_TABLE_PREFIX", "clinical_alg")
  cfg$output_root <- demo_env("DSFLOWER_CLINICAL_OUTPUT_ROOT",
                              file.path(getwd(), "dsflower_output", "clinical_algorithm_matrix"))

  datasets <- list(
    load_breast_cancer_wisconsin(),
    load_uci_cleveland(),
    load_pima_diabetes(),
    load_cdc_diabetes()
  )
  datasets <- filter_by_env(datasets, "DSFLOWER_CLINICAL_DATASETS")
  specs <- filter_by_env(clinical_model_specs(), "DSFLOWER_CLINICAL_MODELS")
  datasets <- Filter(function(dataset) {
    length(specs_for_dataset_profile(specs, dataset$id, cfg$profile)) > 0L
  }, datasets)
  if (!length(datasets)) {
    stop("No dataset/model combinations are applicable for profile ", cfg$profile, call. = FALSE)
  }

  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(cfg$output_root, timestamp)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  dataset_results <- lapply(datasets, run_clinical_dataset, specs = specs, cfg = cfg)
  names(dataset_results) <- vapply(datasets, `[[`, character(1), "id")

  flat <- do.call(c, lapply(dataset_results, function(x) x$results))
  summary <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    demo_id = "clinical_algorithm_matrix",
    description = paste0(
      "Three-site dsFlower benchmark matrix on public clinical datasets, ",
      "comparing centralized and federated held-out performance."
    ),
    privacy_profile = cfg$profile,
    n_sites = length(cfg$urls),
    datasets = lapply(dataset_results, function(x) x[setdiff(names(x), "results")]),
    model_specs = lapply(specs, function(x) x[setdiff(names(x), c("model", "central"))]),
    results = flat
  )

  evidence_file <- demo_env(
    "DSFLOWER_CLINICAL_EVIDENCE_FILE",
    file.path("inst", "extdata", "dsflower_clinical_algorithm_results.json")
  )

  jsonlite::write_json(summary, file.path(out_dir, "summary.json"),
                       auto_unbox = TRUE, pretty = TRUE, na = "null")
  dir.create(dirname(evidence_file), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(summary, evidence_file,
                       auto_unbox = TRUE, pretty = TRUE, na = "null")

  cat("\nClinical algorithm matrix written to:\n")
  cat(normalizePath(file.path(out_dir, "summary.json")), "\n")
  cat(normalizePath(evidence_file), "\n")
  cat("PASS\n")
  invisible(summary)
}

if (isTRUE(getOption("dsflower.benchmark_clinical_algorithms.autorun", TRUE))) {
  run_clinical_algorithm_matrix()
}
