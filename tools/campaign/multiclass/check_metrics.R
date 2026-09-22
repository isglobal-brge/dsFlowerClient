#!/usr/bin/env Rscript
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "multiclass.R"))
y <- rep(1:3, 2)
perfect <- diag(3)[y, ]
m <- ctg_metrics(y, perfect)
stopifnot(m$macro_auc == 1, m$acc == 1, m$logloss == 0)
m <- ctg_metrics(y, matrix(1 / 3, nrow = 6, ncol = 3))
stopifnot(m$macro_auc == 0.5, m$acc == 1 / 3,
          abs(m$logloss - log(3)) < 1e-12)
p <- rbind(c(.6,.3,.1), c(.4,.5,.1), c(.2,.2,.6),
           c(.4,.5,.1), c(.4,.5,.1), c(.2,.3,.5))
# Independent pairwise definition validates average-tie rank AUC.
pairwise <- mean(vapply(1:3, function(k) {
  d <- outer(p[y == k, k], p[y != k, k], "-")
  mean((d > 0) + 0.5 * (d == 0))
}, numeric(1)))
m <- ctg_metrics(y, p)
stopifnot(abs(m$macro_auc - pairwise) < 1e-12,
          abs(m$logloss + mean(log(c(.6,.5,.6,.4,.5,.5)))) < 1e-12,
          m$acc == 5 / 6)
frame <- data.frame(a = c(-2, 0, 4), b = c(4, 6, 8))
z <- ctg_transform(frame, c("a", "b"), list(lower = c(-1, 4), upper = c(1, 8)))
stopifnot(identical(z$a, c(-1, 0, 1)), identical(z$b, c(-1, 0, 1)))
cat("Multiclass metrics and bounded transform checks passed.\n")
