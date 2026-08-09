# Comprehensive federated validation on the current ds.flower.fit() API.
#
# Exercises every supported use case end-to-end against a live federation:
#   * tabular neural   : logreg / mlp / linear-regression / poisson / multiclass
#   * registered ext.  : a rich-vocabulary nn spec (layernorm/gelu/dropout/
#                        silu/leaky_relu) added via ds.flower.register_model()
#   * vision RESOURCE  : resnet18 + densenet121 over an Opal image resource
#   * vision TABLE     : resnet18 + densenet121 over a samples table with a
#                        relative_path column + dsflower.image_data_root
#
# No credentials are hard-coded -- everything comes from DSFLOWER_OPAL_* env vars
# (see lib/benchmark_utils.R), e.g. for the live workshop federation:
#   DSFLOWER_OPAL_URLS=https://nairobi.datashield.live,https://dakar.datashield.live,https://douala.datashield.live \
#   DSFLOWER_OPAL_USERS=administrator DSFLOWER_OPAL_PASSWORDS=password \
#   Rscript validate_federated.R
#
# DP NOTE: DP is enforced server-side with a fixed administrator-owned
# per-training epsilon/delta contract. Repeating a semantically identical
# training deterministically reproduces its protected release; prior trainings
# neither consume persistent state nor change admission for a later training.

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))
suppressMessages(library(dsFlowerClient))

cfg <- demo_config("validate_federated", default_rounds = 1L)
n_sites <- length(cfg$urls)
img_root <- demo_env("DSFLOWER_VAL_IMAGE_ROOT", "/srv/imaging/synth3/images")
img_table <- demo_env("DSFLOWER_VAL_IMAGE_TABLE", "dsflower_demo.synth3_site")
tab_table <- demo_env("DSFLOWER_VAL_TABULAR_TABLE", "dsflower_demo.breast_cancer")
multi_table <- demo_env("DSFLOWER_VAL_MULTICLASS_TABLE", "dsflower_demo.digits")
img_resource <- demo_env("DSFLOWER_VAL_IMAGE_RESOURCE", "dsflower_demo.pathmnist2d")

# Operator pre-step: point image_data_root at the table-image dataset.
for (i in seq_len(n_sites)) {
  opal <- opal_login(cfg, i)
  tryCatch(opalr::dsadmin.set_option(opal, "dsflower.image_data_root", img_root, profile = "default"),
           error = function(e) NULL)
  opalr::opal.logout(opal)
}

builder <- DSI::newDSLoginBuilder()
for (i in seq_len(n_sites))
  builder$append(server = paste0("opal", i), url = cfg$urls[[i]],
                 user = cfg$users[[i]], password = cfg$passwords[[i]], driver = "OpalDriver")
message("Connecting to ", n_sites, " Opal nodes...")
conns <- DSI::datashield.login(builder$build())
on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

results <- list()
record <- function(label, expr) {
  t0 <- Sys.time()
  st <- tryCatch({
    f <- force(expr)
    if (is.null(f)) "NULL" else as.character(f$status)
  }, error = function(e) paste0("ERR:", substr(conditionMessage(e), 1, 55)))
  dt <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  results[[label]] <<- st
  cat(sprintf("[%s] %-32s status=%s (%ss)\n", format(Sys.time(), "%H:%M:%S"), label, st, dt))
}
# Symbols do not alter the server-owned per-training privacy policy.
fit_tab <- function(label, sym, table, model, target, params = list(),
                    target_bounds = NULL) {
  DSI::datashield.assign.table(conns, sym, table)
  feats <- setdiff(dsBaseClient::ds.colnames(sym)[[1]], target)
  record(label, ds.flower.fit(conns, symbol = sym, target = target, features = feats,
                              model = model, model_params = params,
                              target_bounds = target_bounds,
                              rounds = cfg$rounds, verbose = FALSE))
}

message(">>> TABULAR neural (", tab_table, ")")
fit_tab("logreg (bce)",            "BC_lr",  tab_table, "pytorch_logreg",            "malignant")
fit_tab("mlp[64,32] (bce)",        "BC_mlp", tab_table, "pytorch_mlp",               "malignant", list(hidden_layers = c(64L, 32L)))
fit_tab("regression (mse)",        "BC_reg", tab_table, "pytorch_linear_regression", "mean_area",
        target_bounds = list(lower = 0, upper = 5000))
fit_tab("poisson (poisson_nll)",   "BC_poi", tab_table, "pytorch_poisson",           "mean_smoothness",
        target_bounds = list(lower = 0, upper = 1))

message(">>> registered rich-vocabulary extension (layernorm/gelu/dropout/silu/leaky_relu)")
ds.flower.register_model("rich_vocab", "neural",
  generate = function(p) list(kind = "sequential", layers = list(
    list(op = "linear", out = 32L), list(op = "layernorm"), list(op = "gelu"),
    list(op = "dropout", p = 0.2), list(op = "linear", out = 16L), list(op = "silu"),
    list(op = "leaky_relu", negative_slope = 0.05), list(op = "linear", out = "@out"))),
  loss = "bce_logits", parameter_types = character(),
  description = "rich-vocabulary spec validation", overwrite = TRUE)
fit_tab("rich-vocab extension (bce)", "BC_rv", tab_table, "rich_vocab", "malignant")

message(">>> TABULAR multiclass (", multi_table, ", 10-class)")
fit_tab("multiclass (ce, 10)",     "DG", multi_table, "pytorch_multiclass", "digit", list(n_classes = 10L))

message(">>> NEW suite: GLM (svm/negbin/gamma/ordinal), penalized, conv (cnn/tcn)")
fit_tab("svm (hinge)",            "BC_svm", tab_table,   "pytorch_svm",        "malignant")
fit_tab("negbin (count)",         "DG_nb",  multi_table, "pytorch_negbin",     "digit",       list(nb_dispersion = 1.5),
        target_bounds = list(lower = 0, upper = 9))
fit_tab("gamma (positive)",       "BC_gam", tab_table,   "pytorch_gamma",      "mean_radius", list(gamma_shape = 2.0),
        target_bounds = list(lower = 1e-6, upper = 100))
fit_tab("ordinal (CORN, 10)",     "DG_ord", multi_table, "pytorch_ordinal",    "digit",       list(n_classes = 10L))
fit_tab("ridge (L2)",             "BC_rdg", tab_table,   "pytorch_ridge",      "mean_area",   list(weight_decay = 1.0),
        target_bounds = list(lower = 0, upper = 5000))
fit_tab("lasso (L1)",             "BC_lso", tab_table,   "pytorch_lasso",      "mean_area",   list(l1_penalty = 0.05),
        target_bounds = list(lower = 0, upper = 5000))
fit_tab("elasticnet (L1+L2)",     "BC_ent", tab_table,   "pytorch_elasticnet", "mean_area",   list(weight_decay = 1.0, l1_penalty = 0.05),
        target_bounds = list(lower = 0, upper = 5000))
fit_tab("cnn 2D (8x8)",           "DG_cnn", multi_table, "pytorch_cnn",        "digit",       list(input_shape = c(1L, 8L, 8L), n_classes = 10L))
fit_tab("tcn (dilated conv1d)",   "DG_tcn", multi_table, "pytorch_tcn",        "digit",       list(input_shape = c(1L, 64L), n_classes = 10L))

message(">>> IMAGE resource (", img_resource, ")")
record("vision resnet18 RESOURCE",    ds.flower.fit(conns, resource = img_resource, target = "label",
  model = "pytorch_resnet18",    model_params = list(n_classes = 9L, image_size = 224L), rounds = cfg$rounds, verbose = FALSE))
record("vision densenet121 RESOURCE", ds.flower.fit(conns, resource = img_resource, target = "label",
  model = "pytorch_densenet121", model_params = list(n_classes = 9L, image_size = 224L), rounds = cfg$rounds, verbose = FALSE))

message(">>> IMAGE table (", img_table, ", relative_path)")
DSI::datashield.assign.table(conns, "IMG", img_table)
record("vision resnet18 TABLE",    ds.flower.fit(conns, symbol = "IMG", target = "label",
  model = "pytorch_resnet18",    model_params = list(n_classes = 3L, image_size = 224L), rounds = cfg$rounds, verbose = FALSE))
record("vision densenet121 TABLE", ds.flower.fit(conns, symbol = "IMG", target = "label",
  model = "pytorch_densenet121", model_params = list(n_classes = 3L, image_size = 224L), rounds = cfg$rounds, verbose = FALSE))

cat("\n=== SUMMARY ===\n")
pass <- sum(unlist(results) == "0")
for (k in names(results)) cat(sprintf("  %-32s %s\n", k, results[[k]]))
cat(sprintf("PASSED %d / %d use cases\n", pass, length(results)))
if (pass < length(results)) quit(status = 1L)
cat("PASS\n")
