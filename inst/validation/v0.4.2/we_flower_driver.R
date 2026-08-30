# =========================================================================
# dsFlower / dsFlowerClient v0.4.2 -- chapter-7 worked example
# Heart-disease cohort split across three DSLite sites; exported analyst
# API only. Custodian provisioning in we_flower_scaffold.R (each node pins
# its own privacy contract: epsilon = 1 default, delta = 1e-6).
# =========================================================================

SP <- Sys.getenv("WE_SP")
source(file.path(SP, "dsflower_we", "we_flower_scaffold.R"))
suppressPackageStartupMessages(library(dsFlowerClient))
packageVersion("dsFlower")
packageVersion("dsFlowerClient")

## ---- Custodian side: cohort, vertical of sites, node contracts ----------
cohort <- campaign_load_cohort("breast", file.path(SP, "dsflower_p0", "data_cache"))
split <- campaign_split(cohort$df, 20260830L, n_sites = 3L)
bounds <- campaign_bounds(split$train, cohort$meta$features)
we <- we_setup(split$sites,
               work_dir = file.path(SP, "dsflower_we", "run"),
               venv_root = file.path(SP, "dsflower_p0", "venvs"),
               epsilon = 1, delta = 1e-6)
conns <- we$conns

## ---- [ANALYST API] 1. inspect the vetted model-contract registry --------
models <- ds.flower.list_models()
nrow(models)
head(models, 8)
str(ds.flower.model("pytorch_logreg"), max.level = 2)

## ---- [ANALYST API] 2. one federated training request --------------------
result <- ds.flower.fit(
  conns          = conns,
  symbol         = "D",
  target         = "target",
  features       = cohort$meta$features,
  model          = "pytorch_logreg",
  strategy       = "fedavg",
  rounds         = 5L,
  feature_bounds = bounds,
  target_levels  = c("0", "1"),
  holdout        = 0.2,
  torch_backend  = "cpu",
  output_dir     = file.path(SP, "dsflower_we", "artifact"),
  silent         = TRUE,
  verbose        = FALSE)

## ---- [ANALYST API] 3. the structured, release-only result ---------------
result$available
result$run_id
result$history
str(result$holdout, max.level = 2)
names(result)

## Channel B: the released model scored locally on the public test split.
prob <- ds.flower.predict(result, split$test[, cohort$meta$features],
                          type = "prob")
str(prob)
m <- campaign_metrics(split$test$target,
                      if (is.matrix(prob)) prob[, ncol(prob)] else as.numeric(prob))
round(unlist(m), 4)

we_teardown(we)
sessionInfo()$otherPkgs$dsFlowerClient$Version
