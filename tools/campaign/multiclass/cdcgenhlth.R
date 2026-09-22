# CDC GenHlth helpers; public schema and the frozen cohort/split protocol only.
CDCGENHLTH_SHA256 <- "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
CDCGENHLTH_LEVELS <- as.character(1:5)
CDCGENHLTH_FEATURES <- c(
  "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
  "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
  "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "MentHlth",
  "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income")

cdcgenhlth_bounds <- function() {
  lower <- setNames(rep(0, length(CDCGENHLTH_FEATURES)), CDCGENHLTH_FEATURES)
  upper <- setNames(rep(1, length(CDCGENHLTH_FEATURES)), CDCGENHLTH_FEATURES)
  ranges <- list(BMI = c(0, 100), MentHlth = c(0, 30), PhysHlth = c(0, 30),
                 Age = c(1, 13), Education = c(1, 6), Income = c(1, 8))
  for (name in names(ranges)) {
    lower[name] <- ranges[[name]][1]
    upper[name] <- ranges[[name]][2]
  }
  list(lower = unname(lower), upper = unname(upper))
}

# Copied campaign_split algorithm: stratified 80/20, then class-wise dealing.
cdcgenhlth_split <- function(df, seed) {
  n_sites <- 3L
  set.seed(seed)
  by_class <- split(seq_len(nrow(df)), df$target)
  test_idx <- unlist(lapply(by_class, function(ix) {
    sample(ix, max(1L, round(0.2 * length(ix))))
  }), use.names = FALSE)
  train_idx <- setdiff(seq_len(nrow(df)), test_idx)
  shards <- vector("list", n_sites)
  for (ix in split(train_idx, df$target[train_idx])) {
    ix <- sample(ix)
    for (s in seq_len(n_sites)) {
      shards[[s]] <- c(shards[[s]], ix[seq_along(ix) %% n_sites == (s - 1L)])
    }
  }
  list(train = df[train_idx, , drop = FALSE],
       test = df[test_idx, , drop = FALSE],
       sites = lapply(shards, function(ix) df[sort(ix), , drop = FALSE]))
}

cdcgenhlth_transform <- function(df) {
  bounds <- cdcgenhlth_bounds()
  x <- as.matrix(df[, CDCGENHLTH_FEATURES, drop = FALSE])
  for (j in seq_along(CDCGENHLTH_FEATURES)) {
    x[, j] <- (pmin(pmax(x[, j], bounds$lower[j]), bounds$upper[j]) -
                 (bounds$lower[j] + bounds$upper[j]) / 2) /
                ((bounds$upper[j] - bounds$lower[j]) / 2)
  }
  as.data.frame(x)
}

cdcgenhlth_central_fit <- function(train, seed) {
  set.seed(seed)
  frame <- cdcgenhlth_transform(train)
  frame$target <- factor(train$target, levels = CDCGENHLTH_LEVELS)
  fit <- nnet::multinom(target ~ ., data = frame, decay = 0, maxit = 3000L,
                       reltol = 1e-10, MaxNWts = 10000L, trace = FALSE)
  stopifnot(fit$convergence == 0L)
  fit
}

cdcgenhlth_metrics <- function(y, p) {
  p <- as.matrix(p)
  stopifnot(identical(dim(p), c(length(y), 5L)), all(is.finite(p)),
            all(p >= 0), all(p <= 1), max(abs(rowSums(p) - 1)) < 1e-6,
            all(y %in% 1:5))
  auc <- vapply(1:5, function(k) {
    pos <- y == k
    n1 <- sum(pos)
    n0 <- sum(!pos)
    stopifnot(n1 > 0, n0 > 0)
    ranks <- rank(p[, k], ties.method = "average")
    (sum(ranks[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  }, numeric(1))
  list(macro_auc = mean(auc), acc = mean(max.col(p, ties.method = "first") == y),
       logloss = -mean(log(pmax(p[cbind(seq_along(y), y)], 1e-15))),
       auc_by_class = setNames(as.list(auc), CDCGENHLTH_LEVELS))
}

cdcgenhlth_class_sizes <- function(x) {
  setNames(as.list(as.integer(table(factor(x$target, levels = CDCGENHLTH_LEVELS)))),
           CDCGENHLTH_LEVELS)
}

cdcgenhlth_index_hash <- function(x) {
  digest::digest(paste(rownames(x), collapse = ","), algo = "sha256", serialize = FALSE)
}
