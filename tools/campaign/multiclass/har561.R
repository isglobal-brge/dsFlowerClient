# Official HAR engineered features. Only the caller-selected split is opened.
HAR561_SHA256 <- "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
HAR561_INNER_SHA256 <- "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"
HAR561_FEATURES <- sprintf("feature_%03d", 1:561)
HAR561_LEVELS <- as.character(1:6)

har561_load <- function(cache, split) {
  stopifnot(split %in% c("train", "test"))
  archive <- file.path(cache, "har561_inner.zip")
  stopifnot(identical(digest::digest(file = archive, algo = "sha256"),
                      HAR561_INNER_SHA256))
  read_member <- function(name, classes) {
    con <- unz(archive, paste0("UCI HAR Dataset/", split, "/", name,
                               "_", split, ".txt"), open = "r")
    on.exit(close(con))
    read.table(con, header = FALSE, colClasses = classes)
  }
  x <- read_member("X", "numeric")
  y <- read_member("y", "integer")[[1]]
  subjects <- read_member("subject", "integer")[[1]]
  stopifnot(ncol(x) == 561L, nrow(x) == length(y), length(y) == length(subjects),
            all(is.finite(as.matrix(x))), all(y %in% 1:6),
            all(subjects %in% 1:30))
  names(x) <- HAR561_FEATURES
  x$target <- y
  x$subject <- subjects
  rownames(x) <- paste0(split, "_", seq_len(nrow(x)))
  stopifnot(nrow(x) == if (split == "train") 7352L else 2947L,
            length(unique(subjects)) == if (split == "train") 21L else 9L)
  x
}

har561_sites <- function(train, seed) {
  set.seed(seed)
  ids <- sample(sort(unique(train$subject)))
  site_ids <- split(ids, rep(1:3, each = 7L))
  list(subjects = lapply(site_ids, sort),
       data = lapply(site_ids, function(s) train[train$subject %in% s, , drop = FALSE]))
}

har561_metrics <- function(y, p) {
  p <- as.matrix(p)
  stopifnot(identical(dim(p), c(length(y), 6L)), all(is.finite(p)),
            all(p >= 0), all(p <= 1), max(abs(rowSums(p) - 1)) < 1e-6,
            all(y %in% 1:6))
  auc <- vapply(1:6, function(k) {
    pos <- y == k
    n1 <- sum(pos)
    n0 <- sum(!pos)
    stopifnot(n1 > 0, n0 > 0)
    ranks <- rank(p[, k], ties.method = "average")
    (sum(ranks[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  }, numeric(1))
  list(macro_auc = mean(auc), acc = mean(max.col(p, ties.method = "first") == y),
       logloss = -mean(log(pmax(p[cbind(seq_along(y), y)], 1e-15))),
       auc_by_class = setNames(as.list(auc), HAR561_LEVELS))
}

har561_transform <- function(df) {
  as.data.frame(lapply(df[HAR561_FEATURES], function(x) pmin(pmax(x, -1), 1)))
}

har561_central_fit <- function(train, seed) {
  set.seed(seed)
  frame <- har561_transform(train)
  frame$target <- factor(train$target, levels = HAR561_LEVELS)
  fit <- nnet::multinom(target ~ ., data = frame, decay = 0, maxit = 3000L,
                       reltol = 1e-10, MaxNWts = 10000L, trace = FALSE)
  stopifnot(fit$convergence == 0L)
  fit
}

har561_class_sizes <- function(x) {
  setNames(as.list(as.integer(table(factor(x$target, levels = HAR561_LEVELS)))),
           HAR561_LEVELS)
}

har561_index_hash <- function(x) {
  digest::digest(paste(rownames(x), collapse = ","), algo = "sha256", serialize = FALSE)
}
