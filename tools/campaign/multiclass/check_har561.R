#!/usr/bin/env Rscript
# Synthetic checks only: no official test members are opened.
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "har561.R"))
y <- rep(1:6, each = 3L)
p <- diag(6)[y, ]
m <- har561_metrics(y, p)
stopifnot(m$macro_auc == 1, m$acc == 1, m$logloss == 0)
p <- matrix(1 / 6, nrow = length(y), ncol = 6L)
m <- har561_metrics(y, p)
stopifnot(m$macro_auc == 0.5, m$acc == 1 / 6,
          isTRUE(all.equal(m$logloss, log(6))))
set.seed(913)
p <- matrix(sample(1:4, length(y) * 6, replace = TRUE), ncol = 6L)
p <- p / rowSums(p)
pairwise <- vapply(1:6, function(k) {
  differences <- outer(p[y == k, k], p[y != k, k], "-")
  mean((differences > 0) + 0.5 * (differences == 0))
}, numeric(1))
stopifnot(isTRUE(all.equal(har561_metrics(y, p)$macro_auc, mean(pairwise))))
synthetic <- data.frame(subject = rep(1:21, each = 2L), target = rep(1:6, 7L))
sites <- har561_sites(synthetic, 20260820L)
stopifnot(all(lengths(sites$subjects) == 7L),
          identical(sort(unname(unlist(sites$subjects))), 1:21),
          sum(vapply(sites$data, nrow, integer(1))) == nrow(synthetic),
          identical(sites, har561_sites(synthetic, 20260820L)))
cat("HAR561 synthetic metric and subject partition checks passed\n")
