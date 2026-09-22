# CTG data and evaluation helpers; no model or privacy mechanisms are changed.
CTG_FEATURES <- c("LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
                  "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
                  "Mode", "Mean", "Median", "Variance", "Tendency")
CTG_SHA256 <- "4648b5bf338d18f0c7e030a5d2cfb2018c831695cdb90129912cf00367c7d751"
ctg_load <- function(cache) {
  path <- file.path(cache, "cardiotocography.csv")
  stopifnot(identical(digest::digest(file = path, algo = "sha256"), CTG_SHA256))
  raw <- read.csv(path, check.names = FALSE)
  stopifnot(nrow(raw) == 2126L, all(c(CTG_FEATURES, "NSP") %in% names(raw)))
  df <- raw[, CTG_FEATURES]
  df$target <- as.integer(raw$NSP)
  stopifnot(all(complete.cases(df)), all(df$target %in% 1:3),
            all(vapply(df, is.numeric, logical(1))))
  df
}

ctg_metrics <- function(y, p) {
  p <- as.matrix(p)
  stopifnot(identical(dim(p), c(length(y), 3L)), all(is.finite(p)),
            all(p >= 0), all(p <= 1), max(abs(rowSums(p) - 1)) < 1e-6,
            all(y %in% 1:3))
  auc <- vapply(1:3, function(k) {
    pos <- y == k
    n1 <- sum(pos)
    n0 <- sum(!pos)
    stopifnot(n1 > 0, n0 > 0)
    ranks <- rank(p[, k], ties.method = "average")
    (sum(ranks[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  }, numeric(1))
  list(macro_auc = mean(auc), acc = mean(max.col(p, ties.method = "first") == y),
       logloss = -mean(log(pmax(p[cbind(seq_along(y), y)], 1e-15))),
       auc_by_class = setNames(as.list(auc), c("1", "2", "3")))
}

ctg_transform <- function(df, features, bounds) {
  x <- as.matrix(df[, features, drop = FALSE])
  for (j in seq_along(features)) {
    x[, j] <- (pmin(pmax(x[, j], bounds$lower[j]), bounds$upper[j]) -
                 (bounds$lower[j] + bounds$upper[j]) / 2) /
                ((bounds$upper[j] - bounds$lower[j]) / 2)
  }
  as.data.frame(x)
}

ctg_central_fit <- function(train, features, bounds, seed) {
  set.seed(seed)
  frame <- ctg_transform(train, features, bounds)
  frame$target <- factor(train$target, levels = 1:3)
  fit <- nnet::multinom(target ~ ., data = frame, decay = 0, maxit = 3000L,
                       reltol = 1e-10, MaxNWts = 10000L, trace = FALSE)
  stopifnot(fit$convergence == 0L)
  fit
}

ctg_score_federated <- function(fit, test, features) {
  p <- ds.flower.predict(fit, test[, features, drop = FALSE], type = "prob")
  p <- as.matrix(p)
  # The neural predictor emits columns in the declared target-level order.
  if (!is.null(colnames(p)) && all(c("1", "2", "3") %in% colnames(p))) {
    p <- p[, c("1", "2", "3"), drop = FALSE]
  }
  ctg_metrics(test$target, p)
}

ctg_trivial <- function(train, test) {
  prevalence <- as.numeric(table(factor(train$target, levels = 1:3))) / nrow(train)
  p <- matrix(rep(prevalence, each = nrow(test)), ncol = 3L)
  c(ctg_metrics(test$target, p),
    list(majority_class = which.max(prevalence), training_prevalence = prevalence,
         probability_policy = "training class frequencies; argmax is majority"))
}
