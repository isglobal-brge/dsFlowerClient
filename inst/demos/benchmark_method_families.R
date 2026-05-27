#!/usr/bin/env Rscript

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

validation_env <- new.env(parent = globalenv())
sys.source(file.path(script_dir(), "validate_methods.R"), envir = validation_env)

validation_fn <- function(name) {
  if (exists(name, envir = validation_env, inherits = FALSE)) {
    return(get(name, envir = validation_env, inherits = FALSE))
  }
  get(name, envir = globalenv(), inherits = TRUE)
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

load_support2_family_data <- function(limit = as.integer(validation_fn("demo_env")("DSFLOWER_SUPPORT2_LIMIT", 3000L)),
                                      seed = as.integer(validation_fn("demo_env")("DSFLOWER_DEMO_SEED", 4242L))) {
  ensure_python_module("ucimlrepo", "ucimlrepo>=0.0.7")
  python <- dsFlowerClient:::.client_python_cmd()
  out <- tempfile(fileext = ".csv")
  code <- paste(
    "import json, numpy as np, pandas as pd",
    "from ucimlrepo import fetch_ucirepo",
    sprintf("out_path = %s", jsonlite::toJSON(out, auto_unbox = TRUE)),
    sprintf("limit = int(%s)", jsonlite::toJSON(limit, auto_unbox = TRUE)),
    sprintf("seed = int(%s)", jsonlite::toJSON(seed, auto_unbox = TRUE)),
    "ds = fetch_ucirepo(id=880)",
    "X = ds.data.features.copy()",
    "Y = ds.data.targets.copy()",
    "df = pd.concat([X, Y], axis=1)",
    "df = df.dropna(subset=['age', 'hday', 'death', 'hospdead', 'diabetes', 'dzclass', 'num.co', 'aps']).reset_index(drop=True)",
    "rng = np.random.default_rng(seed)",
    "if len(df) > limit:",
    "    # Stratify by hospital mortality so all sites retain enough events.",
    "    idx = []",
    "    for cls in sorted(df['hospdead'].dropna().unique()):",
    "        cls_idx = np.where(df['hospdead'].to_numpy() == cls)[0]",
    "        rng.shuffle(cls_idx)",
    "        take = max(1, int(round(limit * len(cls_idx) / len(df))))",
    "        idx.extend(cls_idx[:take].tolist())",
    "    idx = np.array(idx[:limit])",
    "    rng.shuffle(idx)",
    "    df = df.iloc[idx].reset_index(drop=True)",
    "numeric_features = ['age','num.co','edu','scoma','meanbp','wblc','hrt','resp','temp','pafi','alb','bili','crea','sod','ph','glucose','bun','urine','adlp','adls','adlsc']",
    "categorical_features = ['sex','dzgroup','dzclass','income','race','ca','dnr']",
    "features = []",
    "work = pd.DataFrame(index=df.index)",
    "for col in numeric_features:",
    "    if col in df.columns:",
    "        vals = pd.to_numeric(df[col], errors='coerce')",
    "        vals = vals.fillna(vals.median())",
    "        sd = vals.std(ddof=0)",
    "        if not np.isfinite(sd) or sd == 0:",
    "            sd = 1.0",
    "        work[col] = (vals - vals.mean()) / sd",
    "        features.append(col)",
    "cat = pd.DataFrame(index=df.index)",
    "for col in categorical_features:",
    "    if col in df.columns:",
    "        vals = df[col].astype('object').fillna('missing').astype(str)",
    "        dummies = pd.get_dummies(vals, prefix=col, dtype=float)",
    "        cat = pd.concat([cat, dummies], axis=1)",
    "if cat.shape[1] > 0:",
    "    work = pd.concat([work, cat], axis=1)",
    "safe_cols = []",
    "seen = {}",
    "for col in work.columns:",
    "    safe = ''.join(ch if ch.isalnum() else '_' for ch in str(col)).strip('_').lower()",
    "    if not safe:",
    "        safe = 'feature'",
    "    base = safe",
    "    count = seen.get(base, 0)",
    "    if count:",
    "        safe = f'{base}_{count+1}'",
    "    seen[base] = count + 1",
    "    safe_cols.append(safe)",
    "work.columns = safe_cols",
    "features = list(work.columns)",
    "target = pd.DataFrame(index=df.index)",
    "target['outcome'] = pd.to_numeric(df['death'], errors='coerce').fillna(0).astype(int)",
    "aps = pd.to_numeric(df['aps'], errors='coerce').fillna(pd.to_numeric(df['aps'], errors='coerce').median())",
    "target['severity_score'] = (aps - aps.mean()) / (aps.std(ddof=0) if aps.std(ddof=0) else 1.0)",
    "target['comorbidity_count'] = pd.to_numeric(df['num.co'], errors='coerce').fillna(0).clip(lower=0).astype(int)",
    "dz = pd.Categorical(df['dzclass'].astype(str))",
    "target['diagnosis_class'] = dz.codes.astype(int)",
    "target['label_death'] = pd.to_numeric(df['death'], errors='coerce').fillna(0).astype(int)",
    "target['label_hospital_death'] = pd.to_numeric(df['hospdead'], errors='coerce').fillna(0).astype(int)",
    "target['label_diabetes'] = pd.to_numeric(df['diabetes'], errors='coerce').fillna(0).astype(int)",
    "target['time'] = pd.to_numeric(df['hday'], errors='coerce').fillna(1).clip(lower=1).astype(float)",
    "target['event'] = pd.to_numeric(df['hospdead'], errors='coerce').fillna(0).astype(int)",
    "target['event_type'] = 0",
    "event = target['event'].to_numpy() == 1",
    "cancer = df['dzclass'].astype(str).str.contains('Cancer', case=False, na=False).to_numpy()",
    "target.loc[event & cancer, 'event_type'] = 1",
    "target.loc[event & ~cancer, 'event_type'] = 2",
    "out = pd.concat([pd.Series([f'support2_{i+1}' for i in range(len(df))], name='id'), work, target], axis=1)",
    "out.to_csv(out_path, index=False)",
    sep = "\n"
  )
  run <- processx::run(
    python, c("-c", code),
    env = dsFlowerClient:::.client_venv_env(),
    echo = TRUE,
    error_on_status = FALSE,
    timeout = 1800
  )
  if (run$status != 0L) {
    stop("Could not build SUPPORT2 family benchmark dataset:\n",
         run$stderr, call. = FALSE)
  }
  utils::read.csv(out, check.names = FALSE)
}

family_specs <- function(features, n_classes) {
  demo_env <- validation_fn("demo_env")
  demo_bool <- validation_fn("demo_bool")
  common <- list(batch_size = as.integer(demo_env("DSFLOWER_FAMILY_BATCH", 128L)),
                 local_epochs = as.integer(demo_env("DSFLOWER_FAMILY_LOCAL_EPOCHS", 1L)))
  specs <- list(
    list(method = "pytorch_linear_regression", model = "pytorch_linear_regression", task = "regression",
         target = "severity_score", features = features,
         model_params = modifyList(common, list(learning_rate = 0.02)), rounds = 3L,
         family = "continuous outcome regression",
         notes = "SUPPORT2 severity-score regression with MSE loss.",
         acceptance = list(max_loss_ratio = 1.35, max_loss_margin = 0.20)),
    list(method = "pytorch_poisson", model = "pytorch_poisson", task = "regression",
         target = "comorbidity_count", features = features,
         model_params = modifyList(common, list(hidden_layers = "", learning_rate = 0.01)), rounds = 3L,
         family = "count regression",
         notes = "SUPPORT2 comorbidity-count Poisson regression.",
         acceptance = list(max_loss_ratio = 1.50, max_loss_margin = 0.25)),
    list(method = "pytorch_multiclass", model = "pytorch_multiclass", task = "classification",
         target = "diagnosis_class", features = features,
         model_params = modifyList(common, list(
           hidden_layers = demo_env("DSFLOWER_FAMILY_MULTICLASS_HIDDEN", ""),
           n_classes = as.integer(n_classes),
           learning_rate = as.numeric(demo_env("DSFLOWER_FAMILY_MULTICLASS_LR", 0.02))
         )),
         rounds = as.integer(demo_env("DSFLOWER_FAMILY_MULTICLASS_ROUNDS", 3L)),
         family = "multiclass classification",
         notes = "SUPPORT2 diagnosis-class multiclass classifier.",
         acceptance = list(max_loss_ratio = 1.35, max_loss_margin = 0.20)),
    list(method = "pytorch_multilabel", model = "pytorch_multilabel", task = "classification",
         target = c("label_death", "label_hospital_death", "label_diabetes"), features = features,
         model_params = modifyList(common, list(n_labels = 3L, hidden_layers = "16", learning_rate = 0.01)), rounds = 3L,
         family = "multilabel classification",
         notes = "SUPPORT2 simultaneous death, hospital-death and diabetes labels.",
         acceptance = list(max_loss_ratio = 1.40, max_loss_margin = 0.20)),
    list(method = "pytorch_coxph", model = "pytorch_coxph", task = "survival",
         target = c("time", "event"), features = features,
         model_params = modifyList(common, list(learning_rate = 0.005)), rounds = 3L,
         family = "time-to-event survival",
         notes = "SUPPORT2 hospital time-to-event Cox proportional hazards model.",
         acceptance = list(max_loss_ratio = 1.20, max_loss_margin = 0.20))
  )
  if (isTRUE(demo_bool("DSFLOWER_FAMILY_INCLUDE_AFT", FALSE))) {
    specs[[length(specs) + 1L]] <- list(
      method = "pytorch_lognormal_aft", model = "pytorch_lognormal_aft", task = "survival",
         target = c("time", "event"), features = features,
         model_params = modifyList(common, list(learning_rate = 0.005)), rounds = 3L,
         family = "time-to-event survival",
         notes = "SUPPORT2 log-normal accelerated failure-time model.",
      acceptance = list(max_loss_ratio = 1.35, max_loss_margin = 0.25)
    )
  }
  specs
}

selected_family_specs <- function(specs) {
  selection <- validation_fn("demo_env")("DSFLOWER_FAMILY_MODELS", "all")
  if (!nzchar(selection) || identical(tolower(selection), "all")) return(specs)
  wanted <- trimws(strsplit(selection, ",", fixed = TRUE)[[1]])
  specs[vapply(specs, function(x) x$method %in% wanted, logical(1))]
}

evaluate_pytorch_artifact <- function(output_dir, data_csv, spec, python) {
  model_path <- file.path(output_dir, "model.pt")
  if (!file.exists(model_path)) return(NULL)

  payload <- list(
    model = model_path,
    data = data_csv,
    features = spec$features,
    target = spec$target,
    method = spec$method,
    output = tempfile(fileext = ".json")
  )
  payload_json <- tempfile(fileext = ".json")
  jsonlite::write_json(payload, payload_json, auto_unbox = TRUE, pretty = TRUE)
  code <- paste(
    "import json, math",
    "import numpy as np",
    "import pandas as pd",
    "import torch",
    "cfg = json.load(open(r'''%s'''))",
    "df = pd.read_csv(cfg['data'])",
    "X = torch.tensor(df[cfg['features']].to_numpy(dtype='float32'), dtype=torch.float32)",
    "targets = cfg['target'] if isinstance(cfg['target'], list) else [cfg['target']]",
    "ck = torch.load(cfg['model'], map_location='cpu', weights_only=False)",
    "state = ck.get('state_dict', ck)",
    "weights = [state[k].float() for k in sorted(state.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]",
    "def forward_dense(X, weights):",
    "    h = X",
    "    n_pairs = len(weights) // 2",
    "    for i in range(n_pairs):",
    "        W = weights[2*i]",
    "        b = weights[2*i+1]",
    "        h = h @ W.T + b",
    "        if i < n_pairs - 1:",
    "            h = torch.relu(h)",
    "    return h",
    "def bce_loss(logits, y):",
    "    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y).item()",
    "def cox_loss(log_risk, time, event):",
    "    order = torch.argsort(time, descending=True)",
    "    lr = log_risk[order]",
    "    ev = event[order]",
    "    log_cumsum = torch.logcumsumexp(lr, dim=0)",
    "    loss = -torch.sum((lr - log_cumsum) * ev)",
    "    n_events = ev.sum()",
    "    return (loss / n_events.clamp_min(1.0)).item()",
    "def aft_loss(mu, log_scale, log_time, event):",
    "    sigma = torch.exp(log_scale) + 1e-8",
    "    z = (log_time - mu) / sigma",
    "    log_phi = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)",
    "    log_surv = torch.log(0.5 * torch.erfc(z / math.sqrt(2)) + 1e-15)",
    "    ll_uncensored = log_phi - log_scale - log_time",
    "    ll = event * ll_uncensored + (1.0 - event) * log_surv",
    "    return (-ll.mean()).item()",
    "method = cfg['method']",
    "with torch.no_grad():",
    "    if method == 'pytorch_linear_regression':",
    "        y = torch.tensor(df[targets[0]].to_numpy(dtype='float32')).reshape(-1)",
    "        pred = forward_dense(X, weights).reshape(-1)",
    "        loss = torch.nn.functional.mse_loss(pred, y).item()",
    "        acc = None",
    "        metric = 'mse'",
    "    elif method == 'pytorch_poisson':",
    "        y = torch.tensor(df[targets[0]].to_numpy(dtype='float32')).reshape(-1)",
    "        log_rate = forward_dense(X, weights).reshape(-1)",
    "        loss = torch.nn.functional.poisson_nll_loss(log_rate, y, log_input=True, full=False).item()",
    "        acc = None",
    "        metric = 'poisson_nll'",
    "    elif method == 'pytorch_multiclass':",
    "        y = torch.tensor(df[targets[0]].to_numpy(dtype='int64'), dtype=torch.long)",
    "        logits = forward_dense(X, weights)",
    "        loss = torch.nn.functional.cross_entropy(logits, y).item()",
    "        pred = torch.argmax(logits, dim=1)",
    "        acc = float((pred == y).float().mean().item())",
    "        metric = 'cross_entropy'",
    "    elif method == 'pytorch_multilabel':",
    "        y = torch.tensor(df[targets].to_numpy(dtype='float32'), dtype=torch.float32)",
    "        logits = forward_dense(X, weights)",
    "        loss = bce_loss(logits, y)",
    "        pred = (torch.sigmoid(logits) > 0.5).float()",
    "        acc = float((pred == y).float().mean().item())",
    "        metric = 'multilabel_bce'",
    "    elif method == 'pytorch_coxph':",
    "        time = torch.tensor(df[targets[0]].to_numpy(dtype='float32'))",
    "        event = torch.tensor(df[targets[1]].to_numpy(dtype='float32'))",
    "        risk = forward_dense(X, weights).reshape(-1)",
    "        loss = cox_loss(risk, time, event)",
    "        acc = None",
    "        metric = 'cox_partial_likelihood'",
    "    elif method == 'pytorch_lognormal_aft':",
    "        time = torch.tensor(df[targets[0]].to_numpy(dtype='float32'))",
    "        event = torch.tensor(df[targets[1]].to_numpy(dtype='float32'))",
    "        # The AFT template stores log_scale first, then linear weight/bias.",
    "        if len(weights) >= 3 and weights[0].ndim == 0:",
    "            log_scale, W, b = weights[0].reshape(()), weights[1], weights[2]",
    "        else:",
    "            W, b = weights[0], weights[1]",
    "            log_scale = weights[2].reshape(()) if len(weights) > 2 else torch.tensor(0.0)",
    "        mu = X @ W.T + b",
    "        mu = mu.reshape(-1)",
    "        loss = aft_loss(mu, log_scale, torch.log(time + 1e-8), event)",
    "        acc = None",
    "        metric = 'lognormal_aft_nll'",
    "    else:",
    "        raise ValueError(method)",
    "json.dump({'loss': float(loss), 'accuracy': acc, 'metric_name': metric}, open(cfg['output'], 'w'))",
    sep = "\n"
  )
  code <- sprintf(code, payload_json)
  run <- processx::run(
    python, c("-c", code),
    env = dsFlowerClient:::.client_venv_env(),
    error_on_status = FALSE,
    timeout = 600
  )
  if (run$status != 0L) {
    warning("Could not evaluate federated PyTorch artifact for ", spec$method, ":\n", run$stderr)
    return(NULL)
  }
  jsonlite::fromJSON(payload$output, simplifyVector = FALSE)
}

main <- function() {
  demo_env <- validation_fn("demo_env")
  cfg <- validation_fn("demo_config")("method_family_benchmark", default_rounds = 3L)
  cfg$profile <- demo_env(
    "DSFLOWER_FAMILY_PRIVACY_PROFILE",
    demo_env("DSFLOWER_DEMO_PRIVACY_PROFILE", "clinical_default")
  )
  if (cfg$profile %in% c("clinical_update_noise", "high_sensitivity_dp")) {
    cfg$privacy <- ds.flower.privacy(
      cfg$profile,
      epsilon = as.numeric(demo_env("DSFLOWER_FAMILY_DP_EPSILON", 8)),
      delta = as.numeric(demo_env("DSFLOWER_FAMILY_DP_DELTA", 1e-5)),
      clipping_norm = as.numeric(demo_env("DSFLOWER_FAMILY_DP_CLIP", 1))
    )
  } else {
    cfg$privacy <- ds.flower.privacy(cfg$profile)
  }
  cfg$table_prefix <- demo_env(
    "DSFLOWER_FAMILY_TABLE_PREFIX",
    paste0("method_family_", format(Sys.time(), "%Y%m%d%H%M%S"))
  )
  cfg$output_root <- demo_env(
    "DSFLOWER_FAMILY_OUTPUT_ROOT",
    file.path(getwd(), "dsflower_output", "method_family_benchmark")
  )

  data <- load_support2_family_data(limit = as.integer(demo_env("DSFLOWER_SUPPORT2_LIMIT", 3000L)),
                                    seed = cfg$seed)
  target_cols <- c(
    "outcome", "severity_score", "comorbidity_count", "diagnosis_class",
    "label_death", "label_hospital_death", "label_diabetes",
    "time", "event", "event_type"
  )
  features <- setdiff(names(data), c("id", target_cols))
  specs <- selected_family_specs(family_specs(features, n_classes = length(unique(data$diagnosis_class))))
  if (!length(specs)) stop("No family benchmark specs selected.", call. = FALSE)

  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(cfg$output_root, timestamp)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  data_csv <- file.path(out_dir, "support2_family_data.csv")
  utils::write.csv(data, data_csv, row.names = FALSE)

  data$site <- validation_fn("make_stratified_site_labels")(data$event, n_sites = length(cfg$urls), seed = cfg$seed)
  message("Uploading SUPPORT2 family benchmark table to ", length(cfg$urls), " Opal nodes...")
  table_paths <- validation_fn("write_site_tables")(cfg, data)
  conns <- validation_fn("connect_validation_tables")(cfg, table_paths)
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  caps <- tryCatch(DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
                   error = function(e) NULL)
  secagg_supported <- !is.null(caps) && all(vapply(caps, function(x) {
    isTRUE(x$secure_aggregation_supported)
  }, logical(1)))
  policy_meta <- privacy_policy_summary(cfg$profile, caps)

  python <- validation_fn("find_validation_python")()
  baseline_script <- validation_fn("validation_python_script")()
  records <- list()
  for (spec in specs) {
    message("Benchmarking family route ", spec$method, "...")
    central <- validation_fn("run_central_baseline")(
      spec, data_csv, out_dir, python, baseline_script, cfg$seed
    )
    federated <- validation_fn("run_federated_method")(
      spec, conns, cfg$privacy, secagg_supported, data_csv
    )
    artifact_eval <- evaluate_pytorch_artifact(
      federated$output_dir %||% "", data_csv, spec, python
    )
    if (!is.null(artifact_eval)) {
      federated$loss <- artifact_eval$loss
      federated$accuracy <- artifact_eval$accuracy %||% federated$accuracy
    }
    rec <- validation_fn("record_for_spec")(spec, central, federated)
    rec$family <- spec$family
    records[[spec$method]] <- rec
    utils::write.csv(do.call(rbind, records),
                     file.path(out_dir, "method_family_results.csv"),
                     row.names = FALSE)
  }

  results <- do.call(rbind, records)
  rownames(results) <- NULL
  results$secure_aggregation_required <- policy_meta$secure_aggregation_required
  results$dp_required <- policy_meta$dp_required
  results$dp_scope <- policy_meta$dp_scope
  results$privacy_mechanism <- policy_meta$privacy_mechanism
  site_counts <- as.integer(table(factor(data$site, levels = seq_along(cfg$urls))))
  names(site_counts) <- paste0("opal", seq_along(cfg$urls))
  event_counts <- tapply(data$event, data$site, sum)

  summary <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    demo_id = "method_family_benchmark",
    dataset_id = "support2",
    dataset_label = "SUPPORT2 Study",
    dataset_source = "UCI ML Repository SUPPORT2, derived clinical endpoints",
    privacy_profile = cfg$profile,
    privacy_policy = policy_meta,
    n_sites = length(cfg$urls),
    n_total = nrow(data),
    n_features = length(features),
    site_train_n = as.list(site_counts),
    site_event_n = as.list(as.integer(event_counts)),
    opal_urls = cfg$urls,
    table_paths = table_paths,
    secagg_supported = secagg_supported,
    python = python,
    results = results
  )

  result_json <- file.path(out_dir, "method_family_results.json")
  jsonlite::write_json(summary, result_json,
                       auto_unbox = TRUE, pretty = TRUE, na = "null",
                       digits = NA)

  evidence_file <- demo_env(
    "DSFLOWER_FAMILY_EVIDENCE_FILE",
    file.path(getwd(), "inst", "extdata", "dsflower_method_family_results.json")
  )
  dir.create(dirname(evidence_file), recursive = TRUE, showWarnings = FALSE)
  file.copy(result_json, evidence_file, overwrite = TRUE)

  print(results[, c("family", "method", "centralized_loss", "federated_loss",
                    "delta_loss", "acceptable_loss", "validation_status")])
  message("Wrote method-family benchmark results to ", result_json)
  if (any(results$validation_status != "pass")) {
    stop("One or more method-family benchmark routes failed.", call. = FALSE)
  }
  invisible(summary)
}

if (identical(environment(), globalenv())) {
  main()
}
