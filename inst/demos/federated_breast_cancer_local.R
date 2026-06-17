# Self-contained, RUNNABLE federated breast-cancer demo on a LOCAL DSLite mirror.
# Trains a real federated logistic regression across 3 in-process "hospitals"
# (the same 3 partitions used on the live datashield.live federation) and makes
# a prediction -- no servers, no transport, no network. Validated end to end.
#
# Requires: DSLite, dsBase, dsFlower, dsFlowerClient and a local `tor`-free run
# (everything is on localhost). The flwr venv is auto-provisioned on first use.

library(DSLite); library(DSI); library(dsBase); library(dsFlower); library(dsFlowerClient)

# Local trusted validation: no Secure Aggregation, allow loopback connectivity.
options(dsflower.require_secure_aggregation = FALSE,
        dsflower.restrict_connectivity = FALSE)

# --- 3 hospitals, each with its own partition (data never pooled) ------------
mk <- function(csv) { d <- read.csv(csv); d$id <- NULL; d }
feats <- setdiff(colnames(read.csv("bc_part1.csv")), c("id", "malignant"))
cfg   <- DSLite::defaultDSConfiguration(include = c("dsBase", "dsFlower"))
lite1 <- DSLite::newDSLiteServer(tables = list(D = mk("bc_part1.csv")), config = cfg)
lite2 <- DSLite::newDSLiteServer(tables = list(D = mk("bc_part2.csv")), config = cfg)
lite3 <- DSLite::newDSLiteServer(tables = list(D = mk("bc_part3.csv")), config = cfg)
options(datashield.env = environment())

b <- DSI::newDSLoginBuilder()
b$append(server = "nairobi", url = "lite1", driver = "DSLiteDriver", table = "D")
b$append(server = "dakar",   url = "lite2", driver = "DSLiteDriver", table = "D")
b$append(server = "douala",  url = "lite3", driver = "DSLiteDriver", table = "D")
conns <- DSI::datashield.login(b$build(), assign = TRUE, symbol = "D")

# --- Train a federated logistic regression -----------------------------------
fit <- ds.flower.fit(
  conns, symbol = "D",
  target = "malignant", features = feats,
  model = "sklearn_logreg", strategy = "fedavg",
  privacy = "trusted_internal", rounds = 3L
)
print(fit)

# --- Predict on new patients (locally, no server) ----------------------------
new_patients <- mk("bc_part1.csv")[1:3, feats, drop = FALSE]
print(ds.flower.predict(fit, new_patients, type = "prob"))

DSI::datashield.logout(conns)
