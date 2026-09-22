#!/usr/bin/env Rscript
# Synthetic checks only; no source cohort or sealed held-out data are opened.
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "cdcgenhlth.R"))
y <- rep(1:5, each = 3L)
p <- diag(5)[y, ]
m <- cdcgenhlth_metrics(y, p)
stopifnot(m$macro_auc == 1, m$acc == 1, m$logloss == 0)
p <- matrix(1 / 5, nrow = length(y), ncol = 5L)
m <- cdcgenhlth_metrics(y, p)
stopifnot(m$macro_auc == 0.5, m$acc == 1 / 5,
          isTRUE(all.equal(m$logloss, log(5))))
set.seed(913)
p <- matrix(sample(1:4, length(y) * 5, replace = TRUE), ncol = 5L)
p <- p / rowSums(p)
pairwise <- vapply(1:5, function(k) {
  differences <- outer(p[y == k, k], p[y != k, k], "-")
  mean((differences > 0) + 0.5 * (differences == 0))
}, numeric(1))
stopifnot(isTRUE(all.equal(cdcgenhlth_metrics(y, p)$macro_auc, mean(pairwise))))

bounds <- cdcgenhlth_bounds()
synthetic <- as.data.frame(rbind(bounds$lower - 100, bounds$lower,
  (bounds$lower + bounds$upper) / 2, bounds$upper, bounds$upper + 100))
names(synthetic) <- CDCGENHLTH_FEATURES
transformed <- as.matrix(cdcgenhlth_transform(synthetic))
expected <- matrix(rep(c(-1, -1, 0, 1, 1), length(CDCGENHLTH_FEATURES)), nrow = 5L)
stopifnot(isTRUE(all.equal(unname(transformed), expected)),
          length(CDCGENHLTH_FEATURES) == 20L,
          !any(c("Diabetes_binary", "ID", "GenHlth") %in% CDCGENHLTH_FEATURES))

synthetic <- data.frame(target = rep(1:5, c(17L, 23L, 31L, 39L, 45L)))
rownames(synthetic) <- paste0("source_", seq_len(nrow(synthetic)))
split <- cdcgenhlth_split(synthetic, 20260820L)
train_ids <- rownames(split$train)
test_ids <- rownames(split$test)
site_ids <- unlist(lapply(split$sites, rownames), use.names = FALSE)
stopifnot(identical(split, cdcgenhlth_split(synthetic, 20260820L)),
          !length(intersect(train_ids, test_ids)),
          setequal(c(train_ids, test_ids), rownames(synthetic)),
          !anyDuplicated(site_ids), setequal(site_ids, train_ids),
          identical(as.integer(table(split$test$target)),
                    as.integer(round(0.2 * table(synthetic$target)))))
site_classes <- vapply(split$sites, function(x) {
  as.integer(table(factor(x$target, levels = 1:5)))
}, integer(5))
stopifnot(all(apply(site_classes, 1L, function(x) max(x) - min(x)) <= 1L))
cat("CDC GenHlth synthetic metric, public scaling and partition checks passed\n")
